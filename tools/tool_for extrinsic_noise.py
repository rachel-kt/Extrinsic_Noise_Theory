#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:44:04 2025

@author: rachel
"""

import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QPushButton, QFileDialog, QSplitter, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Plot Viewer")
        self.setGeometry(200, 200, 1000, 700)

        self.data_loaded = False
        self.time = None
        self.X = None
        self.Y = None

        splitter = QSplitter(Qt.Horizontal)

        # Left panel with buttons
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # inside __init__ after other buttons
        self.has_distance_cb = QCheckBox("Has Distance Column")
        self.has_distance_cb.setChecked(False)  # default unchecked
        left_layout.addWidget(self.has_distance_cb)

        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self.load_csv)
        left_layout.addWidget(self.load_btn)

        self.plot_mean_btn = QPushButton("Plot Mean")
        self.plot_mean_btn.clicked.connect(self.plot_mean)
        self.plot_mean_btn.setEnabled(False)
        left_layout.addWidget(self.plot_mean_btn)

        self.plot_paired_btn = QPushButton("Plot Cumulative")
        self.plot_paired_btn.clicked.connect(self.plot_paired_unpaired)
        self.plot_paired_btn.setEnabled(False)
        left_layout.addWidget(self.plot_paired_btn)
        
        self.plot_instant_btn = QPushButton("Plot Instantaneous")
        self.plot_instant_btn.clicked.connect(self.plot_instantaneous)
        self.plot_instant_btn.setEnabled(False)
        left_layout.addWidget(self.plot_instant_btn)
        
        self.plot_ccv_btn = QPushButton("Plot Cross-Covariance")
        self.plot_ccv_btn.clicked.connect(self.plot_cross_covariance)
        self.plot_ccv_btn.setEnabled(False)
        left_layout.addWidget(self.plot_ccv_btn)
        
        self.clear_btn = QPushButton("Clear Plot")
        self.clear_btn.clicked.connect(self.clear_plot)
        left_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("Save Plot")
        self.save_btn.clicked.connect(self.save_plot)
        self.save_btn.setEnabled(False)
        left_layout.addWidget(self.save_btn)

        splitter.addWidget(left_panel)

        # Right panel with plot
        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        splitter.addWidget(self.canvas)

        self.setCentralWidget(splitter)

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                # --- choose reader based on extension ---
                if file_path.lower().endswith(".csv"):
                    T = pd.read_csv(file_path, index_col=0, header=0)
                elif file_path.lower().endswith((".xls", ".xlsx")):
                    T = pd.read_excel(file_path, index_col=None, header=0)
                else:
                    raise ValueError("Unsupported file format (must be CSV or Excel)")
                
                if T.shape[1] < 2:
                    raise ValueError("File must contain at least one time column + one data column.")


                self.time = T.iloc[:, 0].to_numpy()
                data = T.iloc[:, 1:].to_numpy()

                if self.has_distance_cb.isChecked():
                    # Extract paired X, Y, D triplets
                    X = data[:, 1::3]
                    Y = data[:, 0::3]
                    D = data[:, 2::3]
                    print(f"Loaded with distances: X={X.shape}, Y={Y.shape}, D={D.shape}")
                    
                    # Symmetrize
                    self.X = np.hstack([X, data[:, 0::3]])
                    self.Y = np.hstack([Y, data[:, 1::3]])
                    self.D = np.hstack([D, D])

                else:
                    # Extract paired X and Y, then symmetrize
                    X = data[:, 1::2]
                    Y = data[:, 0::2]
                    # self.X = np.hstack([X, data[:, 0::2]])
                    # self.Y = np.hstack([Y, data[:, 1::2]])
                    # D = None
                    
                    # Symmetrize
                    self.X = np.hstack([X, data[:, 0::2]])
                    self.Y = np.hstack([Y, data[:, 1::2]])
                    print(f"Loaded without distances: X={X.shape}, Y={Y.shape}")



                self.data_loaded = True
                self.plot_mean_btn.setEnabled(True)
                self.plot_paired_btn.setEnabled(True)
                self.plot_instant_btn.setEnabled(True)
                self.plot_ccv_btn.setEnabled(True)
                self.save_btn.setEnabled(False)

                print(f"Loaded CSV: time={self.time.shape}, X={self.X.shape}, Y={self.Y.shape}")

            except Exception as e:
                print("Error loading CSV:", e)
                
    def clear_plot(self):
        self.canvas.axes.clear()
        self.canvas.figure.clf()   # reset whole figure (removes subplots too)
        self.canvas.draw()
        
    def save_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        if file_path:
            try:
                self.canvas.figure.savefig(file_path)
                print(f"Plot saved to {file_path}")
            except Exception as e:
                print("Error saving plot:", e)

    def meanCrossCovariancePadded(self, data):
        numberOfCells = int(data.shape[1]/2)
        global_mean = np.mean(data, axis=1)
        g_mean = np.mean(global_mean)
        Dmrna_norm = data - g_mean
        denomLen = data.shape[0]
        traces_norm = []
        for jj in range(0, numberOfCells*2, 2):
            traces_norm.append([
                np.concatenate((Dmrna_norm[:, jj], np.zeros(len(Dmrna_norm[:, jj])-1))),
                np.concatenate((Dmrna_norm[:, jj+1], np.zeros(len(Dmrna_norm[:, jj+1])-1)))
            ])
        crossCovPaired = np.mean([np.fft.ifft(np.fft.fft(f1).conj() * np.fft.fft(f2)).real / denomLen for f1, f2 in traces_norm], 0)
        Rred = np.mean([np.fft.ifft(np.fft.fft(f1).conj() * np.fft.fft(f1)).real / denomLen for f1, f2 in traces_norm], 0)
        Rgreen = np.mean([np.fft.ifft(np.fft.fft(f2).conj() * np.fft.fft(f2)).real / denomLen for f1, f2 in traces_norm], 0)
        return crossCovPaired, Rred, Rgreen

    def plot_mean(self):
        if not self.data_loaded:
            return
        try:
            self.save_btn.setEnabled(True)
            data = np.hstack([self.X, self.Y])
            mean_vals = data.mean(axis=1)

            self.canvas.axes.clear()
            self.canvas.axes.plot(self.time, mean_vals, marker="o")
            self.canvas.axes.set_xlabel("Time")
            self.canvas.axes.set_ylabel("Mean of Data")
            self.canvas.axes.set_title("Time vs Mean(Data)")
            self.canvas.draw()
        except Exception as e:
            print("Error plotting mean:", e)

    def plot_paired_unpaired(self):
        if not self.data_loaded:
            return
        try:
            self.save_btn.setEnabled(True)
            X, Y, time = self.X, self.Y, self.time
            
            tmin = 0 * 60
            ind = time > tmin
            X, Y, time = X[ind, :], Y[ind, :], time[ind]

            # --- Paired calculations ---
            mxdiffy = np.mean((X - Y) ** 2, axis=1) / 2
            mxy = np.mean(X * Y, axis=1)
            mx = np.mean(X, axis=1)
            my = np.mean(Y, axis=1)


            n = len(mxdiffy)
            cmxdiffy = np.cumsum(mxdiffy) / np.arange(1, n + 1)
            cmxy = np.cumsum(mxy) / np.arange(1, n + 1)
            cmx = np.cumsum(mx) / np.arange(1, n + 1)
            cmy = np.cumsum(my) / np.arange(1, n + 1)

            ctot = cmxdiffy
            ccell = cmxy - cmx * cmy

            # --- Unpaired ---
            npairs = 1000
            nallele = X.shape[1]
            I, J = np.triu_indices(nallele, k=1)
            pairs = np.vstack([I, J]).T
            perm = np.random.permutation(len(pairs))
            pairs_random = pairs[perm[:npairs]]

            XX, YY = [], []
            for i, j in pairs_random:
                XX.append(X[:, i])
                YY.append(Y[:, j])
            XX, YY = np.array(XX).T, np.array(YY).T

            mxdiffy = np.mean((XX - YY) ** 2, axis=1) / 2
            mxy = np.mean(XX * YY, axis=1)
            mx = np.mean(XX, axis=1)
            my = np.mean(YY, axis=1)


            n = len(mxdiffy)
            smxdiffy = np.cumsum(mxdiffy) / np.arange(1, n + 1)
            smxy = np.cumsum(mxy) / np.arange(1, n + 1)
            smx = np.cumsum(mx) / np.arange(1, n + 1)
            smy = np.cumsum(my) / np.arange(1, n + 1)

            ustot = smxdiffy
            uscell = smxy - smx * smy

            # --- Plot ---
            self.canvas.axes.clear()
            self.canvas.figure.set_size_inches(8, 10)

            # Subplot 1
            self.canvas.figure.clf()
            ax1 = self.canvas.figure.add_subplot(2, 1, 1)
            ax1.plot(time / 60, ctot + ccell, "g")
            ax1.plot(time / 60, ustot + uscell, "gx")
            ax1.plot(time / 60, ccell, "r")
            ax1.plot(time / 60, uscell, "b")
            ax1.legend(["tot paired", "tot unpaired", "paired", "unpaired"])
            ax1.set_title("Cumulative")

            # Subplot 2
            ax2 = self.canvas.figure.add_subplot(2, 1, 2)
            ax2.plot(time / 60, ccell / (ctot + ccell), "r")
            ax2.plot(time / 60, uscell / (ustot + uscell), "b")
            ax2.legend(["paired/tot", "unpaired/tot"])


            self.canvas.draw()

        except Exception as e:
            print("Error plotting paired/unpaired:", e)
            
    def plot_instantaneous(self):
        if not self.data_loaded:
            return
        try:
            self.save_btn.setEnabled(True)
            X, Y, time = self.X, self.Y, self.time
    
            # --- Paired calculations ---
            mxdiffy = np.mean((X - Y) ** 2, axis=1) / 2
            mxy = np.mean(X * Y, axis=1)
            mx = np.mean(X, axis=1)
            my = np.mean(Y, axis=1)
    
            tot = mxdiffy
            cell = mxy - mx * my
    
            # --- Unpaired calculations ---
            npairs = 1000
            nallele = X.shape[1]
            I, J = np.triu_indices(nallele, k=1)
            pairs = np.vstack([I, J]).T
            perm = np.random.permutation(len(pairs))
            pairs_random = pairs[perm[:npairs]]
    
            XX, YY = [], []
            for i, j in pairs_random:
                XX.append(X[:, i])
                YY.append(Y[:, j])
            XX, YY = np.array(XX).T, np.array(YY).T
    
            mxdiffy = np.mean((XX - YY) ** 2, axis=1) / 2
            mxy = np.mean(XX * YY, axis=1)
            mx = np.mean(XX, axis=1)
            my = np.mean(YY, axis=1)
    
            utot = mxdiffy
            ucell = mxy - mx * my
    
            # --- Plot ---
            self.canvas.figure.clf()
            ax1 = self.canvas.figure.add_subplot(2, 1, 1)
            ax1.plot(time / 60, tot + cell, "g")
            ax1.plot(time / 60, utot + ucell, "gx")
            ax1.plot(time / 60, cell, "r")
            ax1.plot(time / 60, ucell, "b")
            ax1.legend(["tot paired", "tot unpaired", "paired", "unpaired"])
            ax1.set_title("Instantaneous")
    
            ax2 = self.canvas.figure.add_subplot(2, 1, 2)
            ax2.plot(time / 60, cell / (tot + cell), "r")
            ax2.plot(time / 60, ucell / (utot + ucell), "b")
            ax2.legend(["paired/tot", "unpaired/tot"])
    
            self.canvas.draw()
    
        except Exception as e:
            print("Error plotting instantaneous:", e)
            
    def plot_cross_covariance(self):
        if not self.data_loaded:
            return
        try:
            self.save_btn.setEnabled(True)
            data = np.hstack([self.X, self.Y])  # use X and Y together
            
            data_random = self.make_random_data_object(npairs=1000)
            ccv_paired, autoR1, autoR2 = self.meanCrossCovariancePadded(data)
            ccv_random, autoRandom1, autoRandom2 = self.meanCrossCovariancePadded(data_random)
    
            stop = len(data)
            self.canvas.figure.clf()
            ax1 = self.canvas.figure.add_subplot(2, 1, 1)
            
            ax1.plot(ccv_paired[:stop], label="Cross-Cov Paired")
            ax1.plot(ccv_random[:stop], label="Cross-Cov un-paired")
            ax1.plot(autoR1[:stop], label="Auto R1")
            ax1.legend()
            ax1.set_title("Cross-Covariance vs Auto-Correlation")
    
            ax2 = self.canvas.figure.add_subplot(2, 1, 2)
            ax2.plot(autoR1[:stop]-ccv_paired[:stop], label="Auto R1 - Cross-Cov")
            ax2.legend()
            ax2.set_title("Difference")
    
            self.canvas.draw()
    
        except Exception as e:
            print("Error plotting cross-covariance:", e)


    def make_random_data_object(self, npairs=1000):
        """
        Generates a 'data_random' object from self.X and self.Y with random pairing.
        Alternating columns: [XX1, YY1, XX2, YY2, ...]
        """
        if not self.data_loaded:
            print("No data loaded!")
            return None
    
        X, Y = self.X, self.Y
        nallele = X.shape[1]
    
        # --- Random pairs ---
        I, J = np.triu_indices(nallele, k=1)
        pairs = np.vstack([I, J]).T
        perm = np.random.permutation(len(pairs))
        pairs_random = pairs[perm[:npairs]]
    
        XX, YY = [], []
        for i, j in pairs_random:
            XX.append(X[:, i])
            YY.append(Y[:, j])
    
        XX = np.array(XX).T  # shape = (timepoints, npairs)
        YY = np.array(YY).T
    
        # --- Combine into alternating columns ---
        data_random = np.empty((XX.shape[0], XX.shape[1]*2))
        data_random[:, 0::2] = XX
        data_random[:, 1::2] = YY
    
        print(f"Random data object created with shape: {data_random.shape}")
        return data_random


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
