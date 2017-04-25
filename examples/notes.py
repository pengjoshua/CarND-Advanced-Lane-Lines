test_img = plt.imread(os.path.join('test_images', 'test4.jpg'))
b_thresholded = threshold(test_img)
img_size = (b_thresholded.shape[1], b_thresholded.shape[0])

corners = np.float32([[588, 446], [691, 446], [1126, 673], [153, 673]])
offset = 200

src = np.float32([corners[0], corners[1], corners[2], corners[3]])
dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

binary_warped = cv2.warpPerspective(b_thresholded, M, img_size)

plt.figure(figsize=(20, 10))
plt.subplot(1,2,1)
plt.axis('off')
plt.title('Binary Thresholded Image')
plt.imshow(b_thresholded, cmap='gray')

plt.subplot(1,2,2)
plt.axis('off')
plt.title('Binary Warped Image')
plt.imshow(binary_warped, cmap='gray')
plt.show()

###

ym_per_pix = 3.0/72.0 # meters per pixel in y dimension
xm_per_pix = 3.7/660.0 # meters per pixel in x dimension
y_eval = 700
midx = 650

y1 = (2*left_fit[0]*y_eval + left_fit[1])*xm_per_pix/ym_per_pix
y2 = 2*left_fit[0]*xm_per_pix/(ym_per_pix*ym_per_pix)

curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
print("Radius of Curvature: %f" % curvature)

Minv = cv2.getPerspectiveTransform(dst, src)
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted_img.shape[1], undistorted_img.shape[0]))

# Combine the result with the original image
result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)

cv2.putText(result,'Radius of Curvature: %.2fm' % curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

x_left_pix = left_fit[0]*(y_eval**2) + left_fit[1]*y_eval + left_fit[2]
x_right_pix = right_fit[0]*(y_eval**2) + right_fit[1]*y_eval + right_fit[2]
position_from_center = ((x_left_pix + x_right_pix)/2 - midx) * xm_per_pix
if position_from_center < 0:
    text = 'left'
else:
    text = 'right'
cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
plt.imshow(result)
plt.show()

###

class Line:
    def __init__(self):
        # if the first frame of video has been processed
        self.first_frame_processed = False
        self.img = None
        self.mse_tolerance = 0.01
        self.left_fit = [np.array([False])]
        self.right_fit = [np.array([False])]
        self.y_eval = 700
        self.midx = 640
        self.ym_per_pix = 3.0/72.0 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/660.0 # meters per pixel in x dimension
        self.curvature = 0


    def update_fits(self, left_fit, right_fit):
        if self.first_frame_processed:
            left_error = ((self.left_fit[0] - left_fit[0]) ** 2).mean(axis=None)
            right_error = ((self.right_fit[0] - right_fit[0]) ** 2).mean(axis=None)
            if left_error < self.mse_tolerance:
                self.left_fit = 0.75 * self.left_fit + 0.25 * left_fit
            if right_error < self.mse_tolerance:
                self.right_fit = 0.75 * self.right_fit + 0.25 * right_fit
        else:
            self.right_fit = right_fit
            self.left_fit = left_fit
        self.update_curvature(self.right_fit)


    def update_curvature(self, fit):
        y1 = (2*fit[0]*self.y_eval + fit[1])*self.xm_per_pix/self.ym_per_pix
        y2 = 2*fit[0]*self.xm_per_pix/(self.ym_per_pix**2)
        curvature = ((1 + y1*y1)**(1.5))/np.absolute(y2)
        if self.first_frame_processed:
            self.curvature = curvature
        elif np.absolute(self.curvature - curvature) < 500:
            self.curvature = 0.75*self.curvature + 0.25*(((1 + y1*y1)**(1.5))/np.absolute(y2))

    def get_position_from_center(self):
        x_left_pix = self.left_fit[0]*(self.y_eval**2) + self.left_fit[1]*self.y_eval + self.left_fit[2]
        x_right_pix = self.right_fit[0]*(self.y_eval**2) + self.right_fit[1]*self.y_eval + self.right_fit[2]
        return ((x_left_pix + x_right_pix)/2.0 - self.midx) * self.xm_per_pix

###

class Pipeline:
    line = None
    M = None
    Minv = None
    mtx = None
    dist = None

    @staticmethod
    def set_values(line, m, minv, mtx, dist):
        Pipeline.line = line
        Pipeline.M = m
        Pipeline.Minv = minv
        Pipeline.mtx = mtx
        Pipeline.dist = dist

    @staticmethod
    def pipeline(img):
        line, M, Minv, mtx, dist = Pipeline.line, Pipeline.M, Pipeline.Minv, Pipeline.mtx, Pipeline.dist
        if (line is None or M is None or Minv is None or mtx is None or dist is None):
            raise NotImplementedError

        img_size = (img.shape[1], img.shape[0])
        img = cv2.undistort(np.copy(img), mtx, dist, None, mtx)
        binary_warped = cv2.warpPerspective(threshold(img), M, img_size)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        nwindows = 9
        window_height = np.int(binary_warped.shape[0]/nwindows)
        margin = 100
        minpix = 50

        if not line.first_frame_processed:
            histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            leftx_current = leftx_base
            rightx_current = rightx_base

            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin

                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)

                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            line.update_fits(left_fit, right_fit)

            line.first_frame_processed = True

        else:
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_fit = line.left_fit
            right_fit = line.right_fit
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]

            # Fit a second order polynomial to each
            line.update_fits(np.polyfit(lefty, leftx, 2), np.polyfit(righty, rightx, 2))
            left_fit = line.left_fit
            right_fit = line.right_fit

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        # Add Radius of Curvature
        cv2.putText(result,'Radius of Curvature: %.2fm' % line.curvature,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        # Add distance from center
        position_from_center = line.get_position_from_center()
        if position_from_center < 0:
            text = 'left'
        else:
            text = 'right'
        cv2.putText(result,'Distance From Center: %.2fm %s' % (np.absolute(position_from_center), text),(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

        return result

line=Line()
Pipeline.set_values(line, M, Minv, mtx, dist)

output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
output_clip = clip1.fl_image(Pipeline.pipeline)
%time output_clip.write_videofile(output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))
