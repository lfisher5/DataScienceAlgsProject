

'''##############################################
# Programmer: Lauren Fisher
# Class: CptS 322-01, Spring 2022
# Programming Assignment #2
# 2/9/22
#
# Description: This program uses lists to replicate existing dataframe functions in pandas
##############################################'''

import copy
import csv
from tabulate import tabulate
import copy


class MyPyTable:
    """Represents a 2D table of data with column names.
    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.
        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).
        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows = 0
        cols = 0
        for _ in self.data:
            rows += 1

        for _ in self.data[0]:
            cols += 1

        return rows, cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.
        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.
        Returns:
            list of obj: 1D list of values in the column
        Notes:
            Raise ValueError on invalid col_identifier
        """
        try:
            col_index = self.column_names.index(col_identifier)
        except ValueError:
            raise ValueError("ValueError exception thrown") from ValueError

        col = []
        for row in self.data:
            val = row[col_index]
            if val == 'NA':
                if include_missing_values:
                    col.append(val)
            else:
                col.append(val)
        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).
        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        for row in self.data:
            for j, _ in enumerate(row):
                try:
                    row[j] = float(row[j])
                except ValueError:
                    pass

        return self

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        self.data = [x for x in self.data if self.data.index(
            x) not in row_indexes_to_drop]

        return self

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.
        Args:
            filename(str): relative path for the CSV file to open and load the contents of.
        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename) as file:
            csvreader = csv.reader(file)
            self.column_names = next(csvreader)
            for row in csvreader:
                self.data.append(row)

        self.convert_to_numeric()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.
        Args:
            filename(str): relative path for the CSV file to save the contents to.
        Notes:
            Use the csv module.
        """
        with open(filename, "w") as outfile:

            for i in range(len(self.column_names)-1):
                string = str(self.column_names[i]) + ','
                outfile.write(string)
            outfile.write(str(self.column_names[-1]) + '\n')

            for row in self.data:
                for i in range(len(row)-1):
                    string = str(row[i]) + ','
                    outfile.write(string)
                outfile.write(str(row[-1]) + '\n')

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.
        Args:
            key_column_names(list of str): column names to use as row keys.
        Returns
            list of int: list of indexes of duplicate rows found
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        col_indxs = []
        rows = []
        set_of_rows = []
        result = []
        for name in key_column_names:
            if name in self.column_names:
                col_indxs.append(self.column_names.index(name))

        for row in self.data:
            vals = []
            for i, _ in enumerate(row):
                if i in col_indxs:
                    vals.append(row[i])
            rows.append(vals)

        for i, element in enumerate(rows):
            if element in set_of_rows:
                result.append(i)
            else:
                set_of_rows.append(element)

        return result

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        for col_index in range(len(self.column_names)):
            self.data = [x for x in self.data if x[col_index] != 'NA']

        return self

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.
        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.get_column(col_name, False)
        col_avg = sum(col) / len(col)
        col_indx = self.column_names.index(col_name)

        for row in self.data:
            if row[col_indx] == 'NA':
                row[col_indx] = col_avg

        return self

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.
        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        stats_table = []
        if self.data == []:
            return MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], stats_table)
        for col in col_names:
            if col in self.column_names:
                column = self.get_column(col)
                attribute = []
                if column != []:
                    attribute = [col]
                    attribute.append(min(column))
                    attribute.append(max(column))
                    attribute.append((max(column) + min(column))/2)
                    attribute.append(sum(column) / len(column))
                    if len(column) % 2 == 0:
                        attribute.append(
                            (sorted(column)[len(column) // 2] +
                             sorted(column)[-1+len(column) // 2])/2)
                    else:
                        attribute.append(sorted(column)[len(column) // 2])

                stats_table.append(attribute)

        return MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], stats_table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the inner joined table.
        """

        inner_join = MyPyTable()

        col_indxs1 = []
        col_indxs2 = []
        ftable1 = []
        ftable2 = []
        for name in key_column_names:
            if name in self.column_names:
                col_indxs1.append(self.column_names.index(name))
            if name in other_table.column_names:
                col_indxs2.append(other_table.column_names.index(name))

        inner_join.column_names = self.column_names + other_table.column_names

        for row in self.data:
            vals = []
            for i, _ in enumerate(row):
                if i in col_indxs1:
                    vals.append(row[i])
            ftable1.append(vals)

        for row in other_table.data:
            vals = []
            for i, _ in enumerate(row):
                if i in col_indxs2:
                    vals.append(row[i])
            ftable2.append(vals)

        for i, row1 in enumerate(ftable1):
            for j, row2 in enumerate(ftable2):
                if sorted(str(row1)) == sorted(str(row2)):
                    nrow = self.data[i] + other_table.data[j]
                    inner_join.data.append(nrow)

        result = []
        vals = []
        for i, element in enumerate(inner_join.column_names):
            if element in vals:
                result.append(i)
            else:
                vals.append(element)

        inner_join.data = [list(x) for x in zip(
            *[d for i, d in enumerate(zip(*inner_join.data)) if i not in result])]
        inner_join.column_names = [
            x for i, x in enumerate(inner_join.column_names) if i not in result]

        return inner_join

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.
        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.
        Returns:
            MyPyTable: the fully outer joined table.
        Notes:
            Pad the attributes with missing values with "NA".
        """
        outer_join = MyPyTable()

        col_indxs1 = []
        col_indxs2 = []
        no_matches = set()

        for name in key_column_names:
            if name in self.column_names:
                col_indxs1.append(self.column_names.index(name))
            if name in other_table.column_names:
                col_indxs2.append(other_table.column_names.index(name))

        outer_join.column_names = self.column_names + other_table.column_names

        for _, row1 in enumerate(self.data):
            no_match = True
            for j, row2 in enumerate(other_table.data):
                match = True
                for k, _ in enumerate(col_indxs1):
                    if row1[col_indxs1[k]] != row2[col_indxs2[k]]:
                        match = False
                if match:
                    no_matches.add(j)
                    no_match = False
                    new_row = row1 + row2
                    outer_join.data.append(new_row)

            if no_match:
                new_row = row1
                nas = ['NA'] * len(other_table.data[0])
                outer_join.data.append(new_row + nas)

        for i, _ in enumerate(other_table.data):
            if i not in no_matches:
                new_row = ['NA'] * len(self.data[0]) + other_table.data[i]
                for k, _ in enumerate(new_row):
                    for j, val in enumerate(col_indxs1):
                        if k == val:
                            new_row[k] = other_table.data[i][col_indxs2[j]]
                outer_join.data.append(new_row)

        result = []
        set_of_rows = []
        for i, element in enumerate(outer_join.column_names):
            if element in set_of_rows:
                result.append(i)
            else:
                set_of_rows.append(element)

        outer_join.data = [list(x) for x in zip(
            *[d for i, d in enumerate(zip(*outer_join.data)) if i not in result])]
        outer_join.column_names = [
            x for i, x in enumerate(outer_join.column_names) if i not in result]

        return outer_join

    def split_col(self, column, split):
        """Splits given column into array by the parameter specified
        Args:
            split (str): key to call split method on for each value in column
            column(str): column name
        Returns:
            None
        """
        col_idx = self.column_names.index(column)
        for i in range(len(self.data)):
            self.data[i][col_idx] = self.data[i][col_idx].split(split)

    def add_split_rows(self, col):
        """adds rows that have been split to bottom of table for individual instances by value
        Args:
            column(str): column name
        Returns:
            None
        """
        col_idx = self.column_names.index(col)
        for i in range(len(self.data)):
            for _ in range(len(self.data[i][col_idx]) - 1):
                split_row = self.data[i][col_idx]
                dup_row = copy.deepcopy(self.data[i])
                dup_row[col_idx] = split_row[0]
                self.data[i][col_idx].pop(0)
                self.data.append(dup_row)

    def drop_cols(self, cols):

        col_idxs = [self.column_names.index(col) for col in cols]

        self.data = [[row[i] for i in range(
            len(row)) if i not in col_idxs] for row in self.data]

        self.column_names = [
            col_name for col_name in self.column_names if col_name not in cols]
