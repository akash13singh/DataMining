import matplotlib.pyplot as plt

colors = ["r","g","b","y","m"]

def symbol( cluster_index, point_type ):
    symbol = colors[cluster_index];
    if point_type == "point":
        symbol = symbol + "o"
    else:
        symbol = symbol + "^"
    return symbol

def plot( data, centers, filename="plot.png" ):

    # Plot data array_x, array_y
    for i in range(len(data)):
        # print data[i].cluster
        plt.plot( data[i].features[0], data[i].features[1], symbol( data[i].cluster, "point" ) )

    # Plot centers
    for i in range(len(centers)):
        plt.plot(
            [ centers[i].features[0] ],  [ centers[i].features[1] ], symbol( i, "center" )
        )

    plt.axis([-50, 40, -50, 50])

    plt.savefig(filename)

# def values_in_dimensions( data ):
#     dim = ( [], [] )
#     for d in data:
#         for i in range(2):
#             dim[i].append( d.features[i] )
#     return dim

def plot_clusters( clusters, centers, cost, filename="plot.png"):

    for c in range(len(clusters)):
        x_values = []
        y_values = []
        for d in clusters[c]:
            x_values.append(d.features[0])
            y_values.append(d.features[1])

        plt.plot( x_values, y_values, symbol( c, "point" ) )

    for c in range(len(centers)):
        center = centers[c]
        plt.plot( [center.features[0]], [center.features[1]], symbol( c, "center" ), mec="white", mew=1 )

    plt.axis([-50, 40, -50, 50])
    plt.annotate("Cost: "+str(cost), (0,30))
    plt.savefig(filename, transparent=False)
    plt.close()
