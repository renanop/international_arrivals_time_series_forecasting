import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def plot(self, data, plot_name, ax=None,figsize=(8,4),x_label=None, y_label=None, title=None, **args):

        created=False
        # Get chart function
        chart = getattr(sns, plot_name)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created = True
        # Generate fig and ax
        # fig, ax = plt.subplots(figsize=figsize)

        # Plot data
        chart(data=data, ax=ax, **args)

        # Miscellaneous
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return (fig if created else None), ax