from numpy import *
import matplotlib.pyplot as plt
import pylab


def compute_error_for_line_given_points(b, m, points):
	#initalize at 0
	totalError = 0

	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]

		totalError += (y - (m * x + b)) **2

	return totalError / float(len(points))

def gradient_decent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m

	for i in range(num_iterations):
		#Update b and m with new b and m by performing gradient step
		b, m = step_gradient(b, m, array(points), learning_rate)
	return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #computing direction
        #using partial derivatives of our error function
        b_gradient += -(2/float(len(points))) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/float(len(points))) * x * (y - ((m_current * x) + b_current))

    #update b and m values using partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]





def run():
	#Collect data
	points = genfromtxt('data.csv', delimiter=',')


	#Define hyperparameters
	#How fast should we converge
	learning_rate = 0.0001
	# y = mx + b
	inital_b = 0
	inital_m = 0
	num_iterations = 1000

	#Train model
	print 'starting gradient decent at b = {0}, m = {1}, error ={2}'.format(inital_b, inital_m, compute_error_for_line_given_points(inital_b, inital_m, points))
	[b, m] = gradient_decent_runner(points, inital_b, inital_m, learning_rate, num_iterations)
	print 'ending point at b = {1}, m = {2}, error = {3}'.format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))


	# Working on ploting, still have slight error
##	x1 = []
##	y1 = []

##	for i in range(0, len(points)):
##		x1.append(points[i, 0])
##		y1.append(points[i, 1])

##	plt.scatter(x1, y1)
##	plt.plot(x1, x1*m + b)
##	plt.show()



if __name__ == '__main__':
	run()