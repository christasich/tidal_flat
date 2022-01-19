import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_results(data, start=0, end=-1, sample_frac=1, save_fig=False, save_path=False):
    plot_data = data.sample(frac = sample_frac)
    ticks = mpl.dates.YearLocator()

    figure, axes = plt.subplots(nrows=6, ncols=1, figsize=(10,15))
    p1 = plot_data[start:end][['water_height']].plot(ax=axes[0]).xaxis.set_visible(False)
    plot_data[start:end][['elev']].plot(ax=axes[0], color='black', ls='--')
    p2 = plot_data[start:end][['conc']].plot(ax=axes[1]).xaxis.set_visible(False)
    p3 = plot_data[start:end][['suspended_sediment']].plot(ax=axes[2]).xaxis.set_visible(False)
    p4 = plot_data[start:end][['incoming_sediment']].plot(ax=axes[3]).xaxis.set_visible(False)
    p5 = plot_data[start:end][['deposited_sediment']].plot(ax=axes[4]).xaxis.set_visible(False)
    p6 = plot_data[start:end][['elev']].plot(ax=axes[5])
    axes[5].xaxis.set_major_locator(ticks)
    axes[5].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
    axes[5].set_xlabel('Year')
    ylabels = ['Height (m)', 'Concentration ($kg \cdot m^{3}$)', 'Suspended ($kg \cdot m^{-2}$)', 'Incoming ($kg \cdot m^{-2}$)', 'Deposited ($kg \cdot m^{-2}$)', 'Elevation (m)']
    count = 0
    for ax in axes:
        ax.margins(x=0)
        ax.set_ylabel(ylabels[count])
        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.get_legend().remove()
        count = count + 1