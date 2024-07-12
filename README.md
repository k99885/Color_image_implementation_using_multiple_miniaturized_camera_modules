# 복수의 소형화 카메라 모듈을 이용한 컬러이미지 구현

특허 출원번호:10-2024-0073597

## Overview
(Color image implementation using multiple miniaturized camera modules)

- **목적**: 소형화 멀티카메라를 활용하여 고해상도의 이미지를 취득하면서 종래의 카메라 모듈보다 상대적으로 작은 물리적 크기에 구현.
- **영상 처리 기술**: CMOS 이미지 센서를 이용한 여러 대의 모노크롬 카메라 모듈로 각각의 파장에 맞는 컬러필터를 통해 고해상도의 이미지를 취득하고, 각 모듈의 이미지를 병합하여 풀컬러 고해상도 영상을 생성.
- **결과**: 병합된 영상은 레드, 그린, 블루 컬러의 개별 이미지 센서에서 취득된 이미지를 결합하여 고해상도의 풀컬러 이미지를 생성하며, 기존의 베이어 패턴 CMOS 센서보다 작은 크기의 카메라 모듈로 구현 가능.

## 사용 기술
- **프로그래밍 언어**: Python
- **영상처리 라이브러리**: OpenCV


## 1. 영상 취득
![소형화 도면1](https://github.com/k99885/Color_image_implementation_using_multiple_miniaturized_camera_modules/assets/157681578/b5135cb7-39b3-4517-8d4b-36831e509ab5)

이 도면에 해당하는 광학 카메라 모듈로 이미지를 취득할 경우 아래와 같은 이미지를 얻게 됩니다.

![rgb_5](https://github.com/k99885/Color_image_implementation_using_multiple_miniaturized_camera_modules/assets/157681578/bbd8b805-6806-4aa9-832c-a0e7092e9976)

왼쪽  : red 의 정보가 담긴 image(100)

가운데: green 의 정보가 담긴 image(101)

오른쪽: blue 의 정보가 담긴 image(102)

-베젤의 방향을 수직방향하고 영상을 취득하였기 때문에 수직방향으로 이미지의 위상차가 존재합니다.

## 2. 영상 보정

취득한 3개의 단일컬러 영상을 하나로 병합하기 위해선 101로 취득한 이미지를 기준으로 100,102의 이미지들을 warpPerspective변환시킨후 100,101,102 를 병합해야합니다.

```
orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
```

따라서 warpPerspective을 위하여 cv2.ORB_create() 함수를 사용하여 orb 객체를 생성 한 뒤 keypoints와 descriptors들을 얻었습니다.

![rgb_key_5](https://github.com/k99885/Color_image_implementation_using_multiple_miniaturized_camera_modules/assets/157681578/b277fd2f-7d56-40ac-b52f-dbd8193c08b4)

```
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = matcher.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x: x.distance)
matches = matches[:300]

matched_image = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

```
 Brute-Force 매처를 사용해 두 이미지의 특징 서술자를 매칭하고 상위 300개의 매칭 결과를 저장하였습니다.

![r,g_aligned_5](https://github.com/k99885/Color_image_implementation_using_multiple_miniaturized_camera_modules/assets/157681578/41881914-252a-445c-babd-7353a343d39f)

100과 101 , 101과 102 두번의 키포인트 매칭을 진행 하였습니다.

```
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)


for i, match in enumerate(matches):
  points1[i, :] = keypoints1[match.queryIdx].pt
  points2[i, :] = keypoints2[match.trainIdx].pt

# 호모그래피 찾기
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

# 호모그래피를 사용하여 이미지 변환
height, width = im1.shape
im1_aligned = cv2.warpPerspective(im1, h, (width, height))
```

매칭된 키포인트들로 호모그래피를 찾고 cv2.warpPerspective을 통하여 101이미지를 기준으로 100,102이미지를 변형 시켰습니다.

![warpP_5](https://github.com/k99885/Color_image_implementation_using_multiple_miniaturized_camera_modules/assets/157681578/ff117cf9-f4f5-4a61-ac51-1a5347135279)

왼쪽   : 100 to 101

오른쪽 : 102 to 101

## 3. 영상 병합

```
r_channel_aligned,matched_image_r_g  = align_images(r_channel_resized, g_channel_resized)
b_channel_aligned,matched_image_b_g  = align_images(b_channel_resized, g_channel_resized)

merged_image = cv2.merge((b_channel_aligned, g_channel_resized, r_channel_aligned))
```
앞서 영상 보정의 단계를 거쳐 r_channel_aligned(100 to 101) ,b_channel_aligned(102 to 101)를 얻었고 기준이미지 101(g_channel_resized) 을  cv2.merge 을 통하여 하나의 영상으로 취득하였습니다.

![aligned_merged_image_g_5](https://github.com/k99885/Color_image_implementation_using_multiple_miniaturized_camera_modules/assets/157681578/4560f6de-9719-43ce-bea7-ed722fcd5e5c)


