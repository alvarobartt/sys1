pub(super) const PARAM_LAYOUT: [(usize, usize); 18] = [
    (0, 12),
    (12, 12),
    (24, 12),
    (36, 8),
    (48, 16),
    (64, 16),
    (112, 128),
    (240, 128),
    (368, 128),
    (496, 128),
    (624, 4),
    (628, 1),
    (629, 1),
    (632, 4),
    (636, 4),
    (640, 4),
    (644, 4),
    (648, 16),
];
pub(super) const SHARED_BYTES: u32 = 230_400;
pub(super) const BLOCK: [u32; 3] = [384, 1, 1];

#[repr(C, align(64))]
pub(super) struct Packed {
    bytes: [u8; 664],
    pub(super) grid: [u32; 3],
}

impl Packed {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        q: u64,
        k: u64,
        v: u64,
        output: u64,
        cumulative: u64,
        tokens: usize,
        batches: usize,
        scale: f32,
    ) -> Result<Self, &'static str> {
        if tokens == 0
            || tokens > i32::MAX as usize
            || batches == 0
            || batches >= i32::MAX as usize
            || scale.to_bits() != 0.0625f32.to_bits()
        {
            return Err("unsupported native FA4 dimensions or scale");
        }

        for (ptr, heads) in [(q, 16u64), (k, 4), (v, 4), (output, 16)] {
            let bytes = (tokens as u64)
                .checked_mul(heads * 256 * 2)
                .ok_or("native FA4 tensor byte overflow")?;
            if ptr == 0
                || ptr & 15 != 0
                || ptr.checked_add(bytes).is_none_or(|end| end > (1u64 << 57))
            {
                return Err("invalid native FA4 tensor address or alignment");
            }
        }
        if cumulative == 0
            || cumulative & 3 != 0
            || cumulative
                .checked_add((batches as u64 + 1) * 4)
                .is_none_or(|end| end > (1u64 << 57))
        {
            return Err("invalid native FA4 cumulative address or alignment");
        }
        let grid = (tokens as u64 + 127 * batches as u64) / 128 * 16;
        if grid == 0 || grid > i32::MAX as u64 {
            return Err("native FA4 grid exceeds CUDA limits");
        }
        let mut p = Self {
            bytes: [0; 664],
            grid: [grid as u32, 1, 1],
        };
        for (offset, heads) in [(0, 16), (12, 4), (24, 4)] {
            p.u32(offset, tokens as u32);
            p.u32(offset + 4, 256);
            p.u32(offset + 8, heads);
        }
        p.u32(36, 256);
        p.u32(40, 16);
        for offset in [48, 64, 648] {
            p.u64(offset, cumulative);
            p.u32(offset + 8, batches as u32 + 1);
        }
        p.descriptor(112, q, tokens as u64, 16);
        p.descriptor(240, k, tokens as u64, 4);
        p.descriptor(368, v, tokens as u64, 4);

        const ADDRESS_MASK: u64 = 0x01ff_ffff_ffff_fff0;
        p.words(
            496,
            [
                output.wrapping_sub(4096u64 << 31) & ADDRESS_MASK,
                (4096u64 << 29) | 0x0024_6532,
                (4096u64 << 29) | 32,
                0,
                0x3fff_ffff_0000_00ff,
                0x7fff_fffe_0000_000f,
                0x3f00_0000_0000_0000,
                0x7f,
            ],
        );

        p.u32(624, (scale * f32::from_bits(0x3fb8_aa3b)).to_bits());
        p.u32(632, tokens as u32);
        p.u32(636, 16);
        p.u32(640, batches as u32);
        p.u32(644, 640);
        Ok(p)
    }

    fn descriptor(&mut self, offset: usize, address: u64, tokens: u64, heads: u64) {
        let stride = heads * 256;
        let big = u64::from(tokens * stride * 2 >= 0x20000) << 21;
        self.words(
            offset,
            [
                address & 0x01ff_ffff_ffff_fff0,
                (stride << 29) | big | 0x46522,
                32,
                0,
                ((tokens - 1) << 32) | 255,
                heads - 1,
                0x3f00_0000_0000_0000,
                if heads == 16 { 0x7f } else { 0x4f },
            ],
        );
    }
    fn words(&mut self, offset: usize, values: [u64; 8]) {
        for (i, v) in values.into_iter().enumerate() {
            self.u64(offset + i * 8, v);
        }
    }
    fn u32(&mut self, offset: usize, value: u32) {
        self.bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    fn u64(&mut self, offset: usize, value: u64) {
        self.bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }
    pub(super) fn bytes(&self) -> &[u8; 664] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn u32(b: &[u8], i: usize) -> u32 {
        u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
    }
    fn u64(b: &[u8], i: usize) -> u64 {
        u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
    }
    #[test]
    #[ignore = "external reference data"]
    fn matches_original_generated_host_defined_bytes() {
        let cases = std::fs::read(crate::assets::get("fa4_driver_abi_cases.bin").unwrap()).unwrap();
        let cases = cases.as_slice();
        assert_eq!(&cases[..8], b"S1FA4ABI");
        let count = u32(cases, 8) as usize;
        let stride = u32(cases, 12) as usize;
        assert_eq!((count, stride, cases.len()), (12, 764, 9184));
        let mut defined = [false; 664];
        for (i, (offset, size)) in PARAM_LAYOUT.into_iter().enumerate() {
            let n = if (6..=9).contains(&i) {
                64
            } else if [4, 5, 17].contains(&i) {
                12
            } else {
                size
            };
            defined[offset..offset + n].fill(true);
        }
        for case in cases[16..].chunks_exact(stride) {
            let p = Packed::new(
                u64(case, 8),
                u64(case, 16),
                u64(case, 24),
                u64(case, 32),
                u64(case, 40),
                u32(case, 0) as usize,
                u32(case, 4) as usize,
                0.0625,
            )
            .unwrap();
            assert_eq!(p.grid, [u32(case, 720), u32(case, 724), u32(case, 728)]);
            assert_eq!(BLOCK, [u32(case, 732), u32(case, 736), u32(case, 740)]);
            assert_eq!(SHARED_BYTES, u32(case, 744));
            assert_eq!(
                [
                    u32(case, 748),
                    u32(case, 752),
                    u32(case, 756),
                    u32(case, 760)
                ],
                [6, 0, 2, 0]
            );
            for (i, used) in defined.into_iter().enumerate() {
                assert_eq!(
                    p.bytes[i],
                    if used { case[56 + i] } else { 0 },
                    "tokens{} byte{i}",
                    u32(case, 0)
                );
            }
        }
    }
    #[test]
    fn rejects_invalid_shapes_scale_and_address_ranges() {
        let make = |t, b, q, c, s| Packed::new(q, 0x20000, 0x30000, 0x40000, c, t, b, s);
        for (t, b) in [
            (0, 1),
            (1, 0),
            (i32::MAX as usize + 1, 1),
            (1, i32::MAX as usize),
        ] {
            assert!(make(t, b, 0x10000, 0x50000, 0.0625).is_err());
        }
        for s in [f32::NAN, f32::INFINITY, 0.0, 0.125] {
            assert!(make(1, 1, 0x10000, 0x50000, s).is_err());
        }
        for q in [0, 0x10001, (1u64 << 57) - 16, u64::MAX] {
            assert!(make(1, 1, q, 0x50000, 0.0625).is_err());
        }
        for c in [0, 0x50001, (1u64 << 57) - 4, u64::MAX] {
            assert!(make(1, 1, 0x10000, c, 0.0625).is_err());
        }
        assert!(
            make(
                i32::MAX as usize,
                i32::MAX as usize - 1,
                0x10000,
                0x50000,
                0.0625
            )
            .is_err()
        );
        assert!(make(1, 1, 0x10010, 0x50004, 0.0625).is_ok());
    }
}
