# References

## Course background

WPI **RBE/CS 549: Computer Vision**, taught by **Prof. Nitin J. Sanket**. The public material covers the geometric estimation ideas used here: https://rbe549.github.io/.

## Methods and data

- Mur-Artal, Montiel, and Tardós, “ORB-SLAM: A Versatile and Accurate Monocular SLAM System.” https://arxiv.org/abs/1502.00956
- Mur-Artal and Tardós, “ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras.” https://arxiv.org/abs/1610.06475
- Campos et al., “ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM.” https://arxiv.org/abs/2007.11898
- Rublee et al., “ORB: An Efficient Alternative to SIFT or SURF.” https://doi.org/10.1109/ICCV.2011.6126544
- KITTI odometry data and evaluation format: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- OpenCV descriptor matching documentation: https://docs.opencv.org/4.x/d3/da1/classcv_1_1BFMatcher.html

## Implementation boundary

The project uses OpenCV's feature, matching, and geometric solvers, and NumPy/SciPy for numerical computation. The project-specific work connects these components through initialization safeguards, persistent landmark state, tracking, local mapping, validated optimization, testing, and offline evaluation. The SLAM papers are conceptual references, not evidence of equivalent robustness or a dependency on their complete software systems. Required third-party notices must be retained if any code is reused.
