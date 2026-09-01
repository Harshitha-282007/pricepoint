import numpy as np

def predict_quantity(current_price, current_quantity,
                     new_price, elasticity):
    return current_quantity * (
        new_price / current_price
    ) ** elasticity