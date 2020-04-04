import os
import sys

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_log_error, mean_squared_error
import matplotlib.pyplot as plt

import logging
from utils.colored_logger import ColoredLogger

#######################################################################################################################
#                                                     SEIR MODEL
#######################################################################################################################


class SEIR:
    def __init__(self, population_size, inial_inf, t_inc, t_inf):

        self.initial_inf = inial_inf
        self.t_inc = t_inc
        self.t_inf = t_inf
        self.population_size = population_size

        # setting logger
        logging.setLoggerClass(ColoredLogger)
        self.logger = logging.getLogger(__name__)
        sys.excepthook = self.logger.excepthook
        self.logger.setLevel(logging.DEBUG)

    # Susceptible equation
    def _dS_dt(self, S, I, R_t, T_inf):
        return -(R_t / T_inf) * I * S

    # Exposed equation
    def _dE_dt(self, S, E, I, R_t, T_inf, T_inc):
        return (R_t / T_inf) * I * S - (T_inc ** -1) * E

    # Infected equation
    def _dI_dt(self, I, E, T_inc, T_inf):
        return (T_inc ** -1) * E - (T_inf ** -1) * I

    # Recovered/Remove/deceased equation
    def _dR_dt(self, I, T_inf):
        return (T_inf ** -1) * I

    def model_system_eqs(self, t, y, R_n):
        """
        Compiling equations
        """
        R_n = R_n(t) if callable(R_n) else R_n
        S, E, I, R = y

        S_out = self._dS_dt(S, I, R_n, self.t_inf)
        E_out = self._dE_dt(S, E, I, R_n, self.t_inf, self.t_inc)
        I_out = self._dI_dt(I, E, self.t_inc, self.t_inf)
        R_out = self._dR_dt(I, self.t_inf)

        return [S_out, E_out, I_out, R_out]

    def solve(self, t_span, R_n):

        # initial values of ODEs system
        s_0 = (self.population_size - self.initial_inf) / self.population_size
        e_0 = 0
        i_0 = self.initial_inf / self.population_size
        r_0 = 0

        sol = solve_ivp(
            self.model_system_eqs,
            [0, t_span],
            [s_0, e_0, i_0, r_0],
            t_eval=np.arange(t_span),
            args=(R_n,),
        )

        self.sol = sol

    def get_decay(self, func, R_0, L=None, k=None):

        if func == "hill":

            def hill_decay(t):
                return R_0 / (1 + (t / L) ** k)

            return hill_decay
        if func == "const":

            def no_decay(t):
                return np.full(np.shape(t), R_0)

            return no_decay
        else:
            raise NotImplementedError("No other decaying functions implemented")

    def _solve_and_eval(
        self,
        x_0,
        y_inf,
        y_fat,
        decay_func="hill",
        optim_days=np.inf,
        return_params="all",
        weight_data=True,
    ):
        # initial guess for decaying parameters
        # self.res = x_0
        self.R_0, self.cfr, self.k, self.L = x_0

        # applying time decaying R_n
        apply_decay = self.get_decay(decay_func, self.R_0, self.L, self.k)
        # getting prediction
        max_days = len(y_inf)
        self.solve(max_days, apply_decay)

        # print(f"Eval res: {res}")

        _, _, inf, rec = self.sol.y
        return self.eval(
            inf,
            rec,
            y_inf,
            y_fat,
            optim_days=optim_days,
            return_params=return_params,
            weight_data=weight_data,
        )

    def plot_decay_curve(self, num_days):
        # getting decaying parameters
        apply_decay = self.get_decay(self.decay_func, self.R_0, self.L, self.k)
        # applying decaying function
        R_t = apply_decay(np.arange(num_days))
        # making graph
        f = plt.figure(figsize=(16, 5))
        ax = f.add_subplot(1, 1, 1)
        ax.plot(R_t, "y", label="R_0")
        plt.title(f"Decay of R_0 using {self.decay_func} func")
        plt.xlabel("Days", fontsize=10)
        plt.ylabel("R_0 value", fontsize=10)
        plt.legend(loc="best")
        plt.show()

    def fit(
        self,
        y_inf,
        y_fat,
        optim_field="avg",
        r_initial=2.2,
        cfr_initial=0.02,
        k_initial=2,
        L_initial=50,
        bounds=((1, 20), (0, 0.15), (1, 4), (1, 100)),
        decay_func="hill",
        optim_days=20,
        method="L-BFGS-B",
    ):
        self.decay_func = decay_func
        # fitting data
        initial_guess = [r_initial, cfr_initial, k_initial, L_initial]
        res = minimize(
            self._solve_and_eval,
            initial_guess,
            bounds=bounds,
            args=(y_inf, y_fat, decay_func, optim_days, optim_field),
            method=method,
        )
        # getting best ODE system solution
        R_n, cfr, k, L = res.x
        self.logger.info(f"Best params:\n R_n: {R_n}\n CFR: {cfr}\n k: {k}\n L: {L}")

        # solving model with best parameters
        res_msle = self._solve_and_eval(
            res.x, y_inf, y_fat, decay_func, optim_days, return_params="all",
        )
        self.logger.info("=============BEST TRAINING MSLE RESULTS=============")
        self.logger.info(f"Train infected MSLE: {res_msle[0]:0.5f}")
        self.logger.info(f"Train fatalities MSLE: {res_msle[1]:0.5f}")
        self.logger.info(f"Train average MSLE: {res_msle[2]:0.5f}")

    def predict(self, num_train_days, forecast_days):

        # applying time decaying R_n
        apply_decay = self.get_decay(self.decay_func, self.R_0, self.L, self.k)
        # getting prediction
        num_days = num_train_days + forecast_days
        self.solve(num_days, apply_decay)
        sus, exp, inf, rec = self.sol.y

        return sus[-forecast_days:], exp[-forecast_days:], inf[-forecast_days:], rec[-forecast_days:]
        #return self.sol.y

    def eval(
        self,
        pred_inf,
        pred_rec,
        y_inf,
        y_fat,
        optim_days=np.inf,
        return_params="all",
        weight_data=True,
    ):
        # sus, exp, inf, rec = self.sol.y

        # infected cases
        inf_pred = np.clip((pred_inf + pred_rec) * self.population_size, 0, np.inf)
        inf_true = y_inf.values

        # fatalities
        fat_pred = np.clip(pred_rec * self.population_size * self.cfr, 0, np.inf)
        fat_true = y_fat.values

        # calculating mean squared log error
        optim_days = min(optim_days, len(y_inf))  # days to optimise for
        # TODO: change weighting function
        if weight_data:
            weights = (
                1 / np.arange(1, optim_days + 1)[::-1]
            )  # weighting data - recent data is more heavily weighted
        else:
            weights = np.ones(optim_days)
        msle_inf = mean_squared_log_error(
            inf_true[-optim_days:], inf_pred[-optim_days:], weights
        )
        msle_fat = mean_squared_log_error(
            fat_true[-optim_days:], fat_pred[-optim_days:], weights
        )
        msle_avg = np.mean([msle_inf, msle_fat])

        if return_params == "inf":
            return msle_inf
        elif return_params == "fat":
            return msle_fat
        elif return_params == "avg":
            return msle_avg
        elif return_params == "all":
            return msle_inf, msle_fat, msle_avg
        else:
            raise ValueError(
                f"{return_params} is not a valid option for return_params argument"
            )

    def plot(self, data_inf, data_fat, num_train_days, title="SEIR model"):
        sus, exp, inf, rec = self.sol.y

        sus_forecast = sus[num_train_days:]
        sus = sus[:num_train_days]
        # splitting exp
        exp_forecast = exp[num_train_days:]
        exp = exp[:num_train_days]
        # splitting inf
        inf_forecast = inf[num_train_days:]
        inf = inf[:num_train_days]
        # splitting rec
        rec_forecast = rec[num_train_days:]
        rec = rec[:num_train_days]

        # SEIR figure
        f = plt.figure(figsize=(16, 5))
        ax = f.add_subplot(1, 3, 1)

        ax.plot(exp, "y", label="Exposed")
        ax.plot(inf, "r", label="Infected")
        ax.plot(rec, "c", label="Recovered/deceased")
        if len(sus_forecast) > 0:
            ax.plot(
                num_train_days + np.arange(len(exp_forecast)),# len(exp) + np.arange(delta_days),
                exp_forecast,
                color="y",
                label="Exposed pred",
                linestyle="--",
            )
            ax.plot(
                num_train_days + np.arange(len(exp_forecast)),# len(exp) + np.arange(delta_days),
                inf_forecast,
                color="r",
                label="Infected pred",
                linestyle="--",
            )
            ax.plot(
                num_train_days + np.arange(len(exp_forecast)),# len(exp) + np.arange(delta_days),
                rec_forecast,
                color="c",
                label="Recovered/deceased pred",
                linestyle="--",
            )
        plt.title(title)
        plt.xlabel("Days", fontsize=10)
        plt.ylabel("Fraction of population", fontsize=10)
        plt.legend(loc="best")

        # Pred and actual inf cases
        ax2 = f.add_subplot(1, 3, 2)
        # if forecast_days:
        inf_total = np.concatenate([inf, inf_forecast])
        rec_total = np.concatenate([rec, rec_forecast])
        preds = np.clip((inf_total + rec_total) * self.population_size, 0, np.inf)
        ax2.plot(
            num_train_days + np.arange(len(inf_forecast)),
            preds[num_train_days:],
            label="Predict InfectedCases",
            color="b",
            linestyle="--",
        )

        ax2.plot(
            range(num_train_days),
            preds[: num_train_days],
            label="Fit InfectedCases",
            color="b",
        )
        ax2.plot(range(len(data_inf)), data_inf, label="Confirmed InfectedCases", color="orange")
        plt.title("Model predict infected and data")
        plt.ylabel("Num people", fontsize=10)
        plt.xlabel("Days", fontsize=10)
        plt.legend(loc="best")

        # Pred and actual fat cases
        ax3 = f.add_subplot(1, 3, 3)
        # if forecast_days:
        rec_total = np.concatenate([rec, rec_forecast])
        preds_fat = np.clip(rec_total * self.population_size * self.cfr, 0, np.inf)
        ax3.plot(
            num_train_days + np.arange(len(inf_forecast)),
            preds_fat[num_train_days:],
            label="Predict FatalityCases",
            color="b",
            linestyle="--",
        )
        # else:
        #     preds_fat = np.clip(rec * self.population_size * self.cfr, 0, np.inf)
        ax3.plot(
            range(num_train_days), preds_fat[: num_train_days], label="Fit FatalityCases",
        )
        ax3.plot(range(len(data_fat)), data_fat, label="Confirmed FatalityCases")
        plt.title("Model predict fatalities and data")
        plt.ylabel("Num people", fontsize=10)
        plt.xlabel("Days", fontsize=10)

        plt.legend(loc="best")
        plt.show()


