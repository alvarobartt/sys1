Each server instance exposes one model. Its ID comes from `--model-id` or
`--served-model-name`. See [supported models](models/index.md) for checkpoint
IDs.

```bash
curl http://localhost:3000/v1/models
```

```json
{
  "data": [
    {
      "id": "convaiinnovations/laya",
      "object": "model",
      "owned_by": "sys1"
    }
  ]
}
```
