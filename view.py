#!/usr/bin/env python3
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import numpy as np
from qma import QMA, QActions

class SimulatorWindow(QWidget):
    def __init__(self):
        self.qma = QMA(3, 4)
        self.steps = 0

        #  Setup window
        super(SimulatorWindow, self).__init__()
        self.setWindowTitle("QMA Simulator")
        layout = QGridLayout()

        # Setup table view
        self.table = QTableWidget((len(QActions)+1) * self.qma.nodes, 0)
        self.table.setVerticalHeaderLabels([str(action) for node in range(self.qma.nodes) for action in list(QActions)+[""]])
        layout.addWidget(self.table, 0, 0, 1, 4)

        # Setup network parameter inputs
        self.network_box = QGroupBox("Network parameter")
        self.nodes_label = QLabel("nodes:")
        self.nodes_input = QSpinBox()
        self.nodes_input.setValue(self.qma.nodes)
        self.slots_label = QLabel("slots:")
        self.slots_input = QSpinBox()
        self.slots_input.setValue(self.qma.slots)
        network_box_layout = QHBoxLayout()
        network_box_layout.addWidget(self.nodes_label)
        network_box_layout.addWidget(self.nodes_input)
        network_box_layout.addWidget(self.slots_label)
        network_box_layout.addWidget(self.slots_input)
        network_box_layout.addStretch()
        self.network_box.setLayout(network_box_layout)
        layout.addWidget(self.network_box, 1, 0, 1, 4)

        # Setup qma paramter inputs
        self.parameter_box = QGroupBox("QMA parameter")
        self.alpha_label = QLabel("alpha:")
        self.alpha_input = QDoubleSpinBox()
        self.alpha_input.setValue(self.qma.alpha)
        self.gamma_label = QLabel("gamma:")
        self.gamma_input = QDoubleSpinBox()
        self.gamma_input.setValue(self.qma.gamma)
        self.xi_label = QLabel("xi:")
        self.xi_input = QDoubleSpinBox()
        self.xi_input.setValue(self.qma.xi)
        self.rho_label = QLabel("rho:")
        self.rho_input = QDoubleSpinBox()
        self.rho_input.setValue(self.qma.rho)
        parameter_box_layout = QHBoxLayout()
        parameter_box_layout.addWidget(self.alpha_label)
        parameter_box_layout.addWidget(self.alpha_input)
        parameter_box_layout.addWidget(self.gamma_label)
        parameter_box_layout.addWidget(self.gamma_input)
        parameter_box_layout.addWidget(self.xi_label)
        parameter_box_layout.addWidget(self.xi_input)
        parameter_box_layout.addWidget(self.rho_label)
        parameter_box_layout.addWidget(self.rho_input)
        parameter_box_layout.addStretch()
        self.parameter_box.setLayout(parameter_box_layout)
        layout.addWidget(self.parameter_box, 2, 0, 1, 4)

        # setup run and reset buttons
        self.run_box = QGroupBox("Run")
        self.steps_label = QLabel("steps:")
        self.steps_input = QSpinBox()
        self.steps_input.setValue(1)
        self.steps_input.setMaximum(10000) 
        self.run_button = QPushButton('Run', self)
        self.run_button.clicked.connect(self._handle_run)
        self.run_button.setToolTip('Run QMA for the next time steps')
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self._handle_reset)
        self.reset_button.setToolTip('Reset QMA and Q-tables')
        run_layout = QHBoxLayout()
        run_layout.addWidget(self.steps_label)
        run_layout.addWidget(self.steps_input)
        run_layout.addWidget(self.run_button)
        run_layout.addWidget(self.reset_button)
        run_layout.addStretch()
        self.run_box.setLayout(run_layout)
        layout.addWidget(self.run_box, 3, 0, 1, 4)

        self.setLayout(layout)

    def _update_values(self):
        current_frame = int(self.steps / self.qma.slots)
        current_slot = self.steps % self.qma.slots
        qtables, actions, random_actions, rewards = self.qma.get_next_timestep(current_slot)
        qtables = np.round(qtables, 3)

        if current_slot == 0:
            # Start of the new frame -> resize table
            self.table.setColumnCount(self.table.columnCount() + (self.qma.slots+1))
            # add qtable for whole frame
            for node in range(self.qma.nodes):
                for action in range(len(QActions)):
                    for slot in range(self.qma.slots):
                        item = QTableWidgetItem(f"{qtables[node, slot, action]}")
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        self.table.setItem((len(QActions)+1)*node + action, current_frame * (self.qma.slots+1) + slot, item)
        else:
            # add qtable for current slot
            for node in range(self.qma.nodes):
                for action in range(len(QActions)):
                        item = QTableWidgetItem(f"{qtables[node, current_slot, action]}")
                        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                        self.table.setItem((len(QActions)+1)*node + action, current_frame * (self.qma.slots+1) + current_slot, item)
        # add action highlight
        for node in range(self.qma.nodes):
            item = self.table.item((len(QActions)+1)*node + actions[node], current_frame * (self.qma.slots+1) + current_slot)
            item.setBackground(QColor(100, 100, 100))
            item.setText(item.text() + f" ({rewards[node]})")
            if random_actions[node]: 
                item.setText(item.text() + " R") 


    def _handle_reset(self):
        self.qma.alpha = self.alpha_input.value()
        self.qma.gamma = self.gamma_input.value()
        self.qma.xi = self.xi_input.value()
        self.qma.slots = self.slots_input.value()
        self.qma.nodes = self.nodes_input.value()
        self.qma.rho = self.rho_input.value()
        self.qma.reset()
        self.steps = 0
        self.table.setColumnCount(0)
        self.table.setRowCount(self.qma.nodes * (len(QActions)+1))
        self.table.setVerticalHeaderLabels([str(action) for node in range(self.qma.nodes) for action in list(QActions)+[""]])


    def _handle_run(self):
        steps = self.steps_input.value()
        self.qma.alpha = self.alpha_input.value()
        self.qma.gamma = self.gamma_input.value()
        self.qma.xi = self.xi_input.value()
        self.qma.rho = self.rho_input.value()
        for step in range(steps):
            self._update_values()
            self.steps += 1


if __name__ == '__main__':
    print("Welcome to QMA simulator!")
    app = QApplication([])

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

    win = SimulatorWindow()
    win.show()
    app.exec_()
