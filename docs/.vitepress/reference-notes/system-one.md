The `model` field is optional. When provided, it must match the model served by
this instance. `state` may be any JSON value. `questions` must contain at least
one supported `choice`, `score`, or `noul` question.

```bash
curl http://localhost:3000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "convaiinnovations/laya",
    "state": {"message": "I was charged twice. Please refund me today."},
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Where should this ticket go?",
        "criteria": {
          "billing": "payments and refunds",
          "support": "product help"
        }
      }
    }
  }'
```

```json
{
  "model": "convaiinnovations/laya",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.9821,
        "support": 0.0179
      },
      "confidence": 0.8705
    }
  },
  "usage": {
    "input_tokens": 42,
    "output_tokens": 0
  }
}
```

An invalid request returns an error object, for example
`{"error":"questions must not be empty"}`.
