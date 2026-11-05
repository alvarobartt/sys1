fn normalized(probabilities: &[f32]) -> Vec<f64> {
    assert!(
        !probabilities.is_empty(),
        "validated questions have options"
    );
    let total: f64 = probabilities.iter().map(|&p| f64::from(p)).sum();
    if total <= 0. {
        return vec![1. / probabilities.len() as f64; probabilities.len()];
    }
    probabilities
        .iter()
        .map(|&p| f64::from(p) / total)
        .collect()
}

pub(crate) fn choice_confidence(probabilities: &[f32]) -> f32 {
    let n = probabilities.len();
    if n == 1 {
        return 1.;
    }
    let q = normalized(probabilities);
    let top = q.iter().copied().fold(0., f64::max);
    ((n as f64 * top - 1.) / (n - 1) as f64).clamp(0., 1.) as f32
}

pub(crate) fn score_confidence(probabilities: &[f32]) -> f32 {
    let n = probabilities.len();
    if n == 1 {
        return 1.;
    }
    let q = normalized(probabilities);

    let top = q.iter().enumerate().fold(
        0,
        |best, (index, &p)| {
            if p > q[best] { index } else { best }
        },
    );
    let spread: f64 = q
        .iter()
        .enumerate()
        .map(|(i, p)| p * i.abs_diff(top) as f64)
        .sum();
    let midpoint = (n - 1) as f64 / 2.;
    let uniform_spread: f64 = (0..n).map(|i| (i as f64 - midpoint).abs()).sum::<f64>() / n as f64;
    (1. - spread / uniform_spread).max(0.) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_matches_reference_formulas_not_top_probability() {
        assert_eq!(choice_confidence(&[0.5, 0.5]), 0.);
        assert_eq!(choice_confidence(&[0.75, 0.25]), 0.5);
        assert_eq!(choice_confidence(&[0.5, 0.25, 0.25]), 0.25);
        assert_eq!(choice_confidence(&[1.]), 1.);
        assert_eq!(score_confidence(&[1.]), 1.);
        assert_eq!(score_confidence(&[0., 1., 0.]), 1.);
        assert_eq!(score_confidence(&[0.25, 0.5, 0.25]), 0.25);
        assert_eq!(score_confidence(&[0.5, 0.5]), 0.);
    }

    #[test]
    fn confidence_normalizes_mass_and_handles_zero_mass() {
        assert_eq!(choice_confidence(&[3., 1.]), 0.5);
        assert_eq!(choice_confidence(&[0., 0., 0.]), 0.);
        assert_eq!(score_confidence(&[1., 2., 1.]), 0.25);
        assert_eq!(score_confidence(&[0., 0., 0.]), 0.);
    }

    #[test]
    fn score_confidence_retains_first_tied_mode() {
        assert_eq!(score_confidence(&[0.4, 0.4, 0., 0.2]), 0.);
    }
}
