import matplotlib.pyplot as plt
import numpy as np

class LabeledScatterPlot():
    def __init__(self, x_2d, y, products_name, id_range, categories_name):
        self.xy_to_products = {}
        points_idx = 0
        for i in id_range:
            cur_arr = []
            names = np.array([], dtype=object)
            while (points_idx < len(products_name) and y[points_idx] == i):
                cur_arr += [x_2d[points_idx]]
                names = np.append(names, [products_name[points_idx]])
                points_idx += 1

            print(names)
            self.xy_to_products[LabeledScatterPlot.hash_2d(cur_arr)] = names

        fig, ax = plt.subplots()
        for i, cat_name in zip(id_range, categories_name):
            plt.scatter(x_2d[y == i, 0], x_2d[y == i, 1], label=cat_name, picker=True)

        fig.canvas.mpl_connect('pick_event', self.on_pick)
        print("Labeled plot ready")
        plt.legend()
        plt.show()

    def on_pick(self, event):
        xy = event.artist.get_offsets()
        names = self.xy_to_products[LabeledScatterPlot.hash_2d(xy)]
        print("------------")
        print(names[event.ind])

    @staticmethod
    def hash_2d(x_2d):
        res = 0
        mod = 1000000007
        x_pow, y_pow = 1, 1
        for x, y in x_2d:
            x = int(x * 1e7)
            y = int(y * 1e7)
            res = (res + x_pow * x + y_pow * y) % mod
            x_pow = (x_pow * 47) % mod
            y_pow = (y_pow * 51) % mod
        return res
