import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv



x = np.array([1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
y = np.array([0.16, 0.990, 3.095, 4.485, 3.075, 1.010, 0.145])


# plt.plot(x, y, 'o', color='black');
# plt.show()

def calculate_mean_square_approxi(coordinates_X, coordinates_Y):

    D_matrix = np.array(
        [[1 for coordinates_X in range(len(coordinates_X))], coordinates_X, coordinates_X**2])
    D_matrix_transpose = np.transpose(D_matrix)
    left_side_matrix = np.dot(D_matrix, D_matrix_transpose)
    right_side = np.dot(D_matrix, coordinates_Y)
    D_invert = inv(left_side_matrix)
    a_coefficient = np.dot(D_invert, right_side)
    return  a_coefficient


def calculate_least_squares(coordinates_X, coordinates_Y):
    n = len(coordinates_X)
    sum_X = sum(coordinates_X)
    sum_X_2  = sum(coordinates_X**2)
    sum_X_3  = sum(coordinates_X**3)
    sum_X_4 = sum(coordinates_X**4)

    sum_Y = sum(np.log(coordinates_Y))
    sum_Y_multip_x = sum(np.log(coordinates_Y) * coordinates_X)
    sum_Y_multi_x_2 = sum(np.log(coordinates_Y) * coordinates_X**2)

    right_side_matrix_2 = np.transpose([[sum_Y,sum_Y_multip_x ,sum_Y_multi_x_2]])

    left_side_matrix_2 = np.array([
        [n, sum_X, sum_X_2], 
        [sum_X, sum_X_2, sum_X_3],
        [sum_X_2, sum_X_3, sum_X_4]
        ])
    lef_matrix_invert_2 = inv(left_side_matrix_2)
    a_coefficient = np.dot(lef_matrix_invert_2, right_side_matrix_2)
    return a_coefficient
   


def main():
    a_coef_mean_square_approx = calculate_mean_square_approxi(x, y)
    s = np.array(a_coef_mean_square_approx[0] + a_coef_mean_square_approx[1]
                 * x + a_coef_mean_square_approx[2]*x**2)

    a_coef_lest_squares = calculate_least_squares(x, y)
    s_two = np.array(
        np.exp(1*(a_coef_lest_squares[0] + a_coef_lest_squares[1]*x + a_coef_lest_squares[2]*x**2)))

    

    fix,ax = plt.subplots()
    ax.plot(x,s,'b',x,s_two,'r',x, y,'o')
    # ax.plot(x, s_two, x, y, 'o')
    plt.show()


main()
