import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def plot(self, data, plot_name, ax=None,figsize=(8,4),x_label=None, y_label=None, title=None, **args):
        """Utility function that generalizes calls to plotting methods from the seaborn library."""

        # Boolean variable for flow control
        created=False

        # Get chart function from plot name string
        chart = getattr(sns, plot_name)

        # If ax is not provided, the function creates a matplotlib.pyplot.figure object.
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created = True

        # Plot data
        chart(data=data, ax=ax, **args)

        # Setting title and labels
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return (fig if created else None), ax