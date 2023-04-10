from imageprocess.utils.conversions import conv_to_uint8

from typing import List

import numpy as np
import cv2


class ImageAlign:
    """Uses image feature matching to align and warp a set of images of the
    same scene.
    """
    def __init__(
            self,
            feature_method: str = 'akaze',
            detection_threshold: float = 0.001
    ):
        """Initiate class with feature detection method and feature detection
        threshold

        Parameters
        ----------
        feature_method: str, optional
            Image feature detection method. Currently only AKAZE implemented
            (https://docs.opencv.org/4.1.0/d8/d30/classcv_1_1AKAZE.html)
        detection_threshold: float, optional
            Decimal threshold to use to determine "high quality" features,
            lower values result in more features but many may be noisy and could
            result in more false matches during feature matching
        """
        self.feature_method = feature_method
        self.detection_threshold = detection_threshold

    def detect_features(self, images: List[np.array]) -> None:
        """Detect features in all images. If images are not already greyscale
        and uint8, then they will be converted prior to detection

        Parameters
        ----------
        images: List[np.array]
            Original images to align
        """
        self.orig_images = images
        self.keypoints = []
        self.descriptors = []

        for img in self.orig_images:
            if len(img.shape) > 2:
                if img.shape[-1] > 2:
                    print("\t\tConverting to greyscale...")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img = img[:, :, 0]  # ignore mask layer
            if img.dtype != 'uint8':
                print("\t\tConverting to 8-bit...")
                img = conv_to_uint8(img)

            features = self._detect_features(img)

            self.keypoints.append(features['keypoints'])
            self.descriptors.append(features['descriptors'])

    def predict(self, ref_image_index: int) -> None:
        """Predict image transforms that best align each image with the
        selected reference image.

        Parameters
        ----------
        ref_image_index: int
            Input image list index of the image to use as the reference image,
            to which all other images are transformed
        """
        self.ref_image_index = ref_image_index

        self.matches = []
        self.transform_matrices = []
        self.inlier_masks = []
        self.inlier_points = []
        for i in range(len(self.orig_images)):
            if i == ref_image_index:
                self.matches.append([])
                continue
            # find matching features
            matches = self._match_desciptors(
                self.descriptors[i], self.descriptors[ref_image_index]
            )
            self.matches.append(matches)

            # estimate transform
            keypoints = [self.keypoints[i], self.keypoints[ref_image_index]]
            transform_output = self._predict_transform(matches, keypoints)
            self.transform_matrices.append(transform_output['homography'])
            self.inlier_masks.append(transform_output['inlier_mask'])
            self.inlier_points.append(transform_output['inlier_points'])

    def apply(self) -> List[np.array]:
        """Warp images and return as a list"""
        # warp images
        final_images = []
        for i, img in enumerate(self.orig_images):
            if i == self.ref_image_index:
                final_images.append(img)
                continue

            img_warped = cv2.warpPerspective(
                img, self.transform_matrices[i - 1]  #, dsize=mos_shape_xy
            )
            final_images.append(img_warped)
        return final_images

    def _akaze(self, img):
        """Detect keypoints and descriptors using AKAZE method"""
        # AKAZE; G1 finds slightly more KPs than G2
        finder = cv2.AKAZE.create(diffusivity=cv2.KAZE_DIFF_PM_G1,
                                  threshold=self.detection_threshold)
        kp, desc = finder.detectAndCompute(img, mask=None)
        return {'keypoints': kp, 'descriptors': desc}

    def _detect_features(self, img):
        """Call specified feature detection method"""
        if self.feature_method == 'akaze':
            result = self._akaze(img)
        else:
            raise ValueError(
                f"Feature detection method invalid: {self.feature_method}"
            )
        return result

    def _match_desciptors(
            self, desc_0, desc_1, method='BruteForce-Hamming', nn_k=2
    ):
        """Finds matching pairs between 2 sets of image feature descriptors.
        Descriptors must be numpy arrays as per the standard opencv feature
        detection output"""
        matches = []
        matcher = cv2.DescriptorMatcher_create(method)
        # nearest neighbour matching
        matches_tmp = matcher.knnMatch(desc_0, desc_1, k=nn_k)
        for m in matches_tmp:
            if m[0].distance < self.match_conf * m[1].distance:
                matches.append(m[0])
        matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance
        if len(matches) == 0:
            raise Exception("No matches found between image pair")
        return matches

    def _predict_transform(self, matches, keypoints):
        """Calculates 3x3 homography matrix that transforms an image onto base
        image given the keypoints and matches

        Parameters
        ----------
        matches : list of cv2.DMatch
            all matches between image pair
        keypoints : list of lists of cv2.KeyPoint
            keypoints detected in each image [ [image1 kps], [image2 kps] ]
        """
        # get keypoints from matches
        pts1 = []
        pts2 = []
        for m in matches:
            pts1.append(keypoints[0][m.queryIdx].pt)
            pts2.append(keypoints[1][m.trainIdx].pt)
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        # predict homography matrix
        homography, inlier_mask = cv2.findHomography(
            pts1, pts2, method=cv2.RANSAC
        )
        results = {
            'homography': homography,
            'inlier_mask': inlier_mask,
            'inlier_points': (pts1, pts2)
        }
        return results
