from os import error
from gs_classes import *

#Variables:


ITERATIONS = 20
EPSILON = 0.0001
SIMPLE_lEVENBERG_MARQUARDT = False
match_landmarks = False
if __name__ == '__main__':
    # Create Graph object
    gs = Graph()
    gs.SIMPLE_lEVENBERG_MARQUARDT = SIMPLE_lEVENBERG_MARQUARDT

    gs.read_from_constraint_edges_files('Data/dlr.mat')
    fig = plt.figure(1)
    gs.plot_graph("seperate data unoptimized", show_constraints = False)
    print("initial error =" )
    print(gs.compute_global_error())
    fig2 = plt.figure(2)
    for i in range(ITERATIONS):
        dx = gs.linearize_and_solve(match_landmarks)
        # show new estimated paths:
        plt.clf()
        gs.plot_graph("seperate data optimized", show_constraints = False)
        plt.draw()
        plt.pause(1)
        # calculate global error
        err = gs.get_error()
        print('---Global error--- iteration = ', i)
        print(err)
        #termination criterion
        if(np.amax(dx)) < EPSILON:
            print("converged with error =", err)
            break

    print('--------------------')
    print('End of optimization')
    fig2 = plt.figure(3)

    gs.plot_graph("seperate data optimized", show_constraints = False)   
    plt.show()

