# Pre-CLIP vs CLIP

Differences between creating embeddings produced by separate image and text models versus those produced by the CLIP model.

## How to Test

```bash
make build
```

### Pre-CLIP

To compute embeddings using a ResNet model for images and a language model for text:

```bash
make run-preclip-cpu
```

### CLIP

To compute embeddings using the CLIP model:

```bash
make run-clip-cpu
```
