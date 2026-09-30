Check readiness after startup:

```bash
curl http://localhost:3000/health
```

```json
{"status":"ok"}
```

The inference worker must be ready for `200 OK`. If it stops, this route
returns `503 Service Unavailable` with `{"status":"unavailable"}`.
