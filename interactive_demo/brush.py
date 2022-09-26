import numpy as np


class Brushstroke:
    """Represents a continuous brushstroke on the image while the user is drawing."""
    def __init__(self, coords, probability, radius, img_shape):
        self._coords = coords   # (x, y) on image
        self.probability = probability
        self.radius = radius
        self.img_h, self.img_w = img_shape
        self._prev_coords = None
        self._fit_line_z = None # Coefficients in ascending order of degree
        # Whether the fit line is function of x or y coordinates relative to the image
        self._fit_reflect_yx = False

    def add_point(self, coords):
        self._prev_coords = self._coords
        self._coords = coords

    def get_new_brush_points(self):
        """
        Fit a polynomial along brush points to estimate the brush stroke.
        Return list of [x,y] vertices along the polynomial between last 2 brush points.
        """
        if self._prev_coords is None:
            return np.array([[self._coords[0], self._coords[1]]])
        elif self._prev_coords == self._coords:
            return np.array([]) # no new points along brushstroke
        fit_xy = np.zeros((3,), dtype=[('x', np.float64), ('y', np.float64)])
        fit_xy[1] = self._prev_coords[::-1] if self._fit_reflect_yx else self._prev_coords
        fit_xy[2] = self._coords[::-1] if self._fit_reflect_yx else self._coords

        def init_y_axis():
            y_upper_bound = self.img_w if self._fit_reflect_yx else self.img_h
            lspace = np.linspace(0, y_upper_bound-1, y_upper_bound)
            return lspace

        lspace = init_y_axis()

        if self._fit_line_z is None:
            self._fit_line_z = np.flip(np.polyfit(fit_xy['x'][1:], fit_xy['y'][1:], 1))
        else:
            reflect_yx = False
            if fit_xy['x'][1] == fit_xy['x'][2]:
                reflect_yx=True
            else:
                slope_at_last_point = self._estimate_slope_at_point(fit_xy['x'][1], fit_xy['y'][1])
                if ((slope_at_last_point > 0 and fit_xy['x'][2] < fit_xy['x'][1]) or
                    (slope_at_last_point < 0 and fit_xy['x'][2] < fit_xy['x'][1])):
                    reflect_yx=True
            # Add point on near x1 along the previous curve as the 3rd point to fit a new quadratic
            fit_xy['x'][0] = fit_xy['x'][1] - 0.01
            fit_xy['y'][0] = self._eval_poly(fit_xy['x'][0])
            if reflect_yx:
                self._fit_reflect_yx = not self._fit_reflect_yx
                lspace = init_y_axis()
                # Reflect x and y coordinates along y=x, so that (x,y) -> (y,x)
                new_x_vals = fit_xy['y'].copy()
                new_y_vals = fit_xy['x'].copy()
                fit_xy['x'], fit_xy['y'] = new_x_vals, new_y_vals

            fit_xy = np.sort(fit_xy)

            # Calculate coefficients
            self._fit_line_z = np.flip(np.polyfit(fit_xy['x'], fit_xy['y'], 2))

        line_fitx = self._eval_poly(lspace)
        vertices = np.array(list(zip(line_fitx, lspace)))

        # Trim range of vertices
        coord_x_index = 1 if self._fit_reflect_yx else 0
        min_x, max_x = sorted((self._prev_coords[coord_x_index], self._coords[coord_x_index]))
        vertices = vertices[min_x:max_x+1]

        if np.min(vertices) < 0 or np.max(vertices) > 959:
            print("Warning: vertices may be out of bounds on y axis")
        if np.min(np.flip(vertices, 1)) < 0 or np.max(np.flip(vertices, 1)) > 959:
            print("Warning: vertices may be out of bounds on x axis")
        if len(vertices) == 0:
            print("Warning: no vertices found")

        # Ensure no pixels are skipped
        new_vertices = []
        for i, (y1, x1) in enumerate(vertices[:-1]):
            y1, x1 = int(round(y1)), int(round(x1))
            y2, x2 = np.round(vertices[i + 1]).astype(int)
            skipped_y_vals = np.arange(y1 + 1, y2)
            for _, skipped_y in enumerate(skipped_y_vals):
                x_left_bound, x_right_bound = x1, x2
                x_estimate, y_estimate = 0, 0
                for _ in range(10):
                    x_estimate = (x_left_bound + x_right_bound) / 2
                    y_estimate = self._eval_poly(x_estimate)
                    if y_estimate > skipped_y:
                        x_right_bound = x_estimate
                    elif y_estimate < skipped_y:
                        x_left_bound = x_estimate
                    else:
                        break
                new_vertices.append([x_estimate, y_estimate])

        # Ensure each row of vertices is in the form [x, y] relative to the image
        vertices = vertices if self._fit_reflect_yx else np.flip(vertices, 1)

        # Add new vertices to the array of vertices
        if len(new_vertices) > 0:
            vertices = np.append(vertices, new_vertices, axis=0)

        # Round vertices and convert to int
        vertices = np.round(vertices).astype(int)

        if np.min(vertices) < 0 or np.max(vertices) > 959:
            print("Warning: vertices may be out of bounds on y axis")
        if np.min(np.flip(vertices, 1)) < 0 or np.max(np.flip(vertices, 1)) > 959:
            print("Warning: vertices may be out of bounds on x axis")

        return vertices

    def _estimate_slope_at_point(self, x, y):
        change_x = 0.001
        change_y = self._eval_poly(x + change_x) - y
        return change_y / change_x

    def _eval_poly(self, x):
        """Evaluate the polynomial at x."""
        return sum([self._fit_line_z[i]*x**i for i in range(len(self._fit_line_z))])
