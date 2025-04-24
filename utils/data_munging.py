import polars as pl

def reorder(self, new_position, col_name):
    neworder=self.columns
    neworder.remove(col_name)
    neworder.insert(new_position,col_name)
    return self.select(neworder)
pl.DataFrame.reorder=reorder
del reorder