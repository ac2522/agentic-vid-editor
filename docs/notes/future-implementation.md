# Future Implementation Notes

## AI Rotoscoping / Background Removal
- Human segmentation: separate human from background in video frames
- May require custom model training
- Research candidates: SAM 2, Robust Video Matting, or similar
- Same modular pattern as other tools: abstract interface with swappable backends
- Scope: after SigLIP 2 + PySceneDetect implementation is complete
