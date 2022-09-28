import numpy as np
from numpy.polynomial import Polynomial as P
import sys

class Brushstroke:
    """Represents a continuous brushstroke on the image while the user is drawing."""
    def __init__(self, probability, radius, img_shape):
        self.probability = probability
        self.radius = radius
        self.img_h, self.img_w = img_shape
        self._coords = None
        self._prev_coords = None
        self._fit_curve_p = None
        self._fit_reflect_yx = False

    def add_point(self, coords):
        """
        Add a new point to the brushstroke.

        Args:
            coords (tuple): (x, y) coordinates of the new point.
        """
        self._prev_coords = self._coords
        self._coords = coords

    def get_new_brush_points(self):
        """
        Fits a polynomial along brush points to estimate the brush stroke.

        Returns:
            numpy.ndarray: array of [x,y] vertices along the brush stroke.
        """

        if self._prev_coords is None:
            return np.array([[self._coords[0], self._coords[1]]]).astype(int)
        if self._prev_coords == self._coords:
            return np.array([]) # no new points along brushstroke

        fit_xy = np.zeros((3,), dtype=[('x', np.float64), ('y', np.float64)])
        fit_xy[1] = self._prev_coords[::-1] if self._fit_reflect_yx else self._prev_coords
        fit_xy[2] = self._coords[::-1] if self._fit_reflect_yx else self._coords

        if self._fit_curve_p is None:
            rank = 1
        else:
            # Add a point near the previous x along the last curve
            x0_x1_change = 0.0001
            fit_xy['x'][0] = fit_xy['x'][1] - x0_x1_change
            fit_xy['y'][0] = self._eval_poly(fit_xy['x'][0])

            if fit_xy['x'][1] == fit_xy['x'][2]:
                # Fit a quadratic between the 3 points only if the previous y is the midpoint y-val
                rank = 2 if fit_xy['y'][1] not in (np.min(fit_xy['y']), np.max(fit_xy['y'])) else 1
            elif np.all(fit_xy['y']==fit_xy['y'][0]):
                # Fit a horizontal line
                rank = 1
            else:
                rank = 2

            if rank == 2:
                # Fit a quadratic to the 3 points
                y0_y1_change = np.subtract(fit_xy['y'][1], fit_xy['y'][0])
                x1_tangent_slope_estimate = np.divide(y0_y1_change, x0_x1_change)
                x1_slope_sign = -1 if x1_tangent_slope_estimate < 0 else 1
                x1_min_slope_magnitude, x1_max_slope_magnitude = 0.01, 100

                if fit_xy['x'][1] == fit_xy['x'][2]:
                    x1_adjusted_slope_magnitude = max(np.absolute(x1_tangent_slope_estimate),
                                                    x1_min_slope_magnitude)
                else:
                    x1_adjusted_slope_magnitude = min(np.absolute(x1_tangent_slope_estimate),
                                                    x1_max_slope_magnitude)

                adjusted_y0_y1_change = x1_slope_sign * x0_x1_change * x1_adjusted_slope_magnitude
                fit_xy['y'][0] = np.subtract(fit_xy['y'][1], adjusted_y0_y1_change)

        if fit_xy['x'][1] == fit_xy['x'][2]:
            self._fit_reflect_yx = not self._fit_reflect_yx
            # Reflect x and y coordinates along y=x, so that (x,y) -> (y,x)
            fit_xy['x'], fit_xy['y'] = fit_xy['y'].copy(), fit_xy['x'].copy()

        # Prepare fit_xy to fit a new curve
        fit_xy = fit_xy[1:] if rank == 1 else fit_xy
        fit_xy = np.sort(fit_xy)

        # Fit the new curve
        self._fit_curve_p = P.fit(fit_xy['x'], fit_xy['y'], rank)

        # Get the domain of the brush stroke (all x integer values between the two original points)
        coord_x_index = 1 if self._fit_reflect_yx else 0
        x_min, x_max = sorted((self._prev_coords[coord_x_index], self._coords[coord_x_index]))
        domain_space = np.linspace(start=x_min, stop=x_max, num=x_max-x_min+1)

        # Evaluate the polynomial at each x value in the domain to get an array of [y,x] vertices
        line_fitx = self._eval_poly(domain_space)
        vertices = np.array(list(zip(line_fitx, domain_space)))

        # Ensure all pixels along the brush stroke are accounted for in the vertices
        new_vertices = []
        for i, (cur_y, cur_x) in enumerate(vertices[:-1]):
            cur_y, cur_x = int(round(cur_y)), int(round(cur_x))
            next_y, next_x = np.round(vertices[i + 1]).astype(int)
            skipped_y_vals = np.arange(cur_y + 1, next_y)
            for _, skipped_y in enumerate(skipped_y_vals):
                x_left_bound, x_right_bound = cur_x, next_x
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
                new_vertices.append([y_estimate, x_estimate])

        # Add new vertices to the array of vertices
        if len(new_vertices) > 0:
            vertices = np.append(vertices, new_vertices, axis=0)

        with open('log.txt', 'a') as file:
            sys.stdout = file
            #print(str(self._fit_curve_p).replace('·', '').replace('¹', '').replace('²', '^2'))
            print("REFLECTED" if self._fit_reflect_yx else "NOT REFLECTED")
            if len(vertices) == 0:
                print(f"Current point: {self._coords}")
            if np.min(vertices) < 0 or np.max(vertices) > 959 or np.min(np.flip(vertices, 1)) < 0 or np.max(np.flip(vertices, 1)) > 959:
                print(f"Warning: vertices may be out of bounds:{vertices}")
            if len(vertices) == 0:
                print("Warning: no vertices found")
            print(f" p_1 = {fit_xy[0]}")
            print(f" p_2 = {fit_xy[1]}")
            if rank == 2:
                print(f" p_3 = {fit_xy[2]}")
                min_x = min(self._coords[0], self._prev_coords[0])
                max_x = max(self._coords[0], self._prev_coords[0])
                min_y = min(self._coords[1], self._prev_coords[1])
                max_y = max(self._coords[1], self._prev_coords[1])
                span_x = abs(max_x - min_x) + 1
                span_y = abs(max_y - min_y) + 1
                x_space = np.linspace(start=min_x, stop=max_x, num=max_x-min_x+1)
                y_space = np.linspace(start=min_y, stop=max_y, num=max_y-min_y+1)
                if self._fit_reflect_yx:
                    min_x, min_y = min_y, min_x
                    max_x, max_y = max_y, max_x
                    x_space, y_space = y_space.copy(), x_space.copy()
                min_x_vert = np.min(vertices[:,1])
                max_x_vert = np.max(vertices[:,1])
                min_y_vert = np.min(vertices[:,0])
                max_y_vert = np.max(vertices[:,0])
                span_x_vert = abs(max_x_vert - min_x_vert) + 1
                span_y_vert = abs(max_y_vert - min_y_vert) + 1
                if max(max(span_x/span_x_vert, span_x_vert/span_x), max(span_y/span_y_vert, span_y_vert/span_y)) > 2:
                    print(f"Warning: span of vertices is too large: \n{vertices}")
                print("*********************************************************")
        sys.stdout = sys.__stdout__

        # Ensure each row of vertices is in the form [x, y] relative to the image
        vertices = vertices if self._fit_reflect_yx else np.flip(vertices, 1)

        # Round vertices and convert to int
        vertices = np.round(vertices).astype(int)

        return vertices

    def _eval_poly(self, x):
        """Evaluate the polynomial at x."""
        return self._fit_curve_p(x)
