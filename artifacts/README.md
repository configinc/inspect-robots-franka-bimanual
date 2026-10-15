# Rollout experiments

All six runs used `openai/gpt-5.6-luna` with low reasoning effort for the same
red → yellow → green → blue block-placement task. Each directory contains the
original Inspect Robots JSON log plus synchronized exterior, left-wrist, and
right-wrist MP4s.

| Run | Control and observation setup | Outcome |
| --- | --- | --- |
| [`adhoc_9b7654b4`](adhoc_9b7654b4/) | Absolute joint position baseline; cameras on demand | Cancelled after 6 joint moves |
| [`adhoc_772ebed0`](adhoc_772ebed0/) | Cartesian EEF delta, 2 cm bounds; cameras on demand | Cancelled after 18 Cartesian moves |
| [`adhoc_9a28788b`](adhoc_9a28788b/) | Selectable absolute joint mode with Y-frame directions | Cancelled after 9 joint moves |
| [`adhoc_ea58a824`](adhoc_ea58a824/) | Joint mode with Franka DH chain and mounting rotations | Gave up without a verified red grasp |
| [`adhoc_5fcb690a`](adhoc_5fcb690a/) | Cartesian EEF delta, 5 cm bounds; cameras on demand | Grasped and carried red, but released it outside the tray |
| [`adhoc_7c583c61`](adhoc_7c583c61/) | Cartesian EEF delta, 5 cm bounds; all three cameras every turn | Cancelled during red alignment |

Inspect a transcript with:

```bash
inspect-robots inspect artifacts/adhoc_5fcb690a/result.json --transcript
```

The MP4 filenames identify their view:

- `scene-0-e0_exterior_cam.mp4` — center/exterior view
- `scene-0-e0_left_wrist_cam.mp4` — left wrist view
- `scene-0-e0_right_wrist_cam.mp4` — right wrist view
