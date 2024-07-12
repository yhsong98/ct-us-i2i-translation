Run the 'CTslice.py' script. This script includes several processing steps:
1. Randomly flip and rotate the 2D slice;
2. Applying a fanshape mask onto the CT slice and the CT slice mask;
3. Assign each class index in the slice mask with a unique color (for compatibility with the ultrasound dataset).

```bash
python CTslice.py --ct_dir ./CTimages --mask_dir ./CTmasks --save_dir ./ProcessedCT --save_dir_mask ./ProcessedMasks
```
