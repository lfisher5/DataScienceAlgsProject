'''##############################################
# Programmer: Lauren Fisher
# Class: CptS 322-01, Spring 2022
# Programming Assignment #3
# 2/22/22
# I attempted the bonus...
#
# Description: This program uses matplotlib to create reusable plotting functions
##############################################'''

import matplotlib.pyplot as plt
import utils


def categorical_frequency_bar_plot(col_name, x, y, x_ticks=None):
    """PLots categorical frequency bar chart for x and y
        Args:
            col_name(str): string for a column name 
            x(list of str): strings for x values
            y(list of int): counts for x values
            x_ticks(boolean): rotation of labels identifier
        Returns:
            None
        """
    plt.figure(figsize=[18, 5])
    plt.bar(x, y)
    plt.title('Distribution of ' + col_name + ' values')
    plt.xlabel(col_name)
    plt.ylabel('Count')

    if x_ticks:
        xtick_labels = x
        plt.xticks(x, xtick_labels, rotation=50, ha="right")

    plt.show()


def categorical_frequency_pie_plot(col_name, x, y, autoPct=True):
    """PLots categorical pie chart for x and y
        Args:
            col_name(str): string for a column name 
            x(list of str): strings for x values
            y(list of int): counts for x values
            autoPct(boolean): pct format identifier
        Returns:
            None
        """
    plt.figure(figsize=[20, 10])
    plt.title('Distribution of ' + col_name)
    if autoPct:
        plt.pie(y, labels=x, autopct="%1.1f%%",
                rotatelabels=True, startangle=20)
    else:
        plt.pie(y, labels=x, rotatelabels=True, startangle=20)

    plt.show()


def plot_hist(table, col_names):
    """PLots histogram of counts with equal binning for columns in table
        Args:
            col_names(list of str): strings for chart identifiers
            table(mypytable): table with data and methods
        Returns:
            None
        """
    for col in col_names:
        x = table.get_column(col)
        plt.figure()
        plt.xlabel(col)
        plt.ylabel('Counts')
        plt.title('Frequency of ' + col)
        plt.hist(x)
        plt.show()


def plot_correlations(x, y, x_names, y_names, regression=False, multiple=True, cov=None):
    """Plots correlation scatter plot for one or multiple datasets
        Args:
            x_names(list of str): names of x values
            y_names(list of str): names of y values
            x(list of str): strings for x values
            y(list of int): counts for x values
            regression(boolean): include regression line on graph
            multiple(boolean): multiple datasets being plotted
            cov(list of list of float): covariance and correl coeff values
        Returns:
            None
        """

    if multiple:
        for i, x_col in enumerate(x):
            for j, y_col in enumerate(y):
                plt.figure(figsize=[15, 5])
                plt.scatter(y_col, x_col)
                plt.title(y_names[j] + ' vs ' + x_names[i])
                if regression:
                    m, b = utils.compute_slope_intercept(y_col, x_col)
                    plt.plot([min(y_col), max(y_col)], [m * min(y_col) +
                                                        b, m * max(y_col) + b], c="r", lw=5)

                    plt.annotate("$y="+str(float(m))+"x+"+str(float(b))+"$", xy=(0.5, 0.75), xycoords="axes fraction",
                                 horizontalalignment="center", color="red")
                if cov:
                    plt.annotate('covariance: ' + str(cov[j][0]) + ', correlation coefficient:' + str(cov[j][1]), xy=(0.5, 0.65), xycoords="axes fraction",
                                 horizontalalignment="center", color="red")

                plt.xlabel(y_names[j])
                plt.ylabel(x_names[i])
                plt.show()
    else:
        plt.figure(figsize=[15, 5])
        plt.scatter(y, x)
        plt.title(y_names + ' vs ' + x_names)
        if regression:
            m, b = utils.compute_slope_intercept(y, x)
            plt.plot([min(y), max(y)], [m * min(y) +
                                        b, m * max(y) + b], c="r", lw=5)

            plt.annotate("$y="+str(float(m))+"x+"+str(float(b))+"$", xy=(0.5, 0.75), xycoords="axes fraction",
                         horizontalalignment="center", color="red")
        if cov:
            plt.annotate('covariance: ' + str(cov[0][0]) + ', correlation coefficient:' + str(cov[0][1]), xy=(0.5, 0.65), xycoords="axes fraction",
                         horizontalalignment="center", color="red")
        plt.xlabel(y_names)
        plt.ylabel(x_names)
        plt.show()


def box_plot_example(distributions, labels, col_names):
    """Plots box plot for multiple labels provided
        Args:
            distributions(list of str): list of values for labels
            col_names(list of str): names of y values
            labels(list of str): strings for x values
        Returns:
            None
        """

    plt.figure(figsize=[30, 10])
    plt.boxplot(distributions)
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[1])
    plt.title('Distributions of ' + col_names[0] + ' by ' + col_names[1])
    plt.xticks(list(range(1, len(distributions) + 1)), labels)

    plt.show()
