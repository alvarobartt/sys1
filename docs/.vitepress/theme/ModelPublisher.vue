<script setup>
import avatars from '../publisher-avatars.json';

const props = defineProps({
  modelId: { type: String, required: true }
});

const publisher = props.modelId.split('/')[0];
const avatarUrl = avatars[publisher] || '';
</script>

<template>
  <a
    class="model-publisher"
    :href="`https://huggingface.co/${encodeURIComponent(publisher)}`"
    target="_blank"
    rel="noopener noreferrer"
  >
    <span class="model-publisher-avatar" aria-hidden="true">
      <img v-if="avatarUrl" :src="avatarUrl" alt="" width="28" height="28" />
      <span v-else>{{ publisher.slice(0, 1).toUpperCase() }}</span>
    </span>
    <span>Published by {{ publisher }}</span>
  </a>
</template>
