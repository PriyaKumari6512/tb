# Project Details

## Overview
This document provides a comprehensive overview of the project's information, including the technologies used, project structure, and components involved in the project.

## Technologies
- **Programming Languages:** Python, JavaScript
- **Frameworks:** Flask, React
- **Databases:** PostgreSQL
- **Tools:** Git, Docker, Jenkins

## Project Structure
```
project-directory/
├── src/
│   ├── components/
│   ├── services/
│   └── utils/
├── tests/
├── docker/
│   └── Dockerfile
├── .gitignore
├── README.md
└── requirements.txt
```

## Components
1. **Frontend:** React components for UI.
2. **Backend:** Flask services for handling business logic and database interactions.
3. **Database:** PostgreSQL for data storage.
4. **Testing:** Unit tests to ensure code quality.
5. **Deployment:** Docker containerization for easy deployment and scaling.





def _load_pretrained(self, path: str):
    """Load pretrained encoder weights (e.g., ImageNet-1K trained MiT)."""
    import os
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pretrained weights file not found: {path}")

    state = torch.load(path, map_location="cpu", weights_only=False)

    # unwrap common checkpoint containers
    if isinstance(state, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "params"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")

    # normalize prefixes to match self.encoder keys
    prefix_candidates = [
        "encoder.",
        "backbone.",
        "model.encoder.",
        "model.backbone.",
        "module.encoder.",
        "module.backbone.",
        "mit.",
        "model.mit.",
        "module.mit.",
    ]

    encoder_state = {}
    enc_keys = set(self.encoder.state_dict().keys())

    for k, v in state.items():
        nk = k
        # strip one matching prefix
        for p in prefix_candidates:
            if nk.startswith(p):
                nk = nk[len(p):]
                break

        # keep only keys that exist in encoder with same shape
        if nk in enc_keys and self.encoder.state_dict()[nk].shape == v.shape:
            encoder_state[nk] = v

    if len(encoder_state) == 0:
        sample = list(state.keys())[:20]
        raise RuntimeError(
            "No compatible encoder keys found in pretrained checkpoint.\n"
            f"Checkpoint: {path}\n"
            f"Sample keys: {sample}"
        )

    missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
    loaded = len(encoder_state)
    print(f"[Pretrained] loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}")
