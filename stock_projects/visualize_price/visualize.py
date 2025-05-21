import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.dates as mdates

class StockVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.5)

        # Labels above input fields
        self.label_stock_ax = plt.axes([0.1, 0.35, 0.2, 0.04])
        self.label_start_ax = plt.axes([0.4, 0.35, 0.2, 0.04])
        self.label_end_ax   = plt.axes([0.7, 0.35, 0.2, 0.04])
        self.label_stock = TextBox(self.label_stock_ax, '', initial='Stock Symbol')
        self.label_start = TextBox(self.label_start_ax, '', initial='Start Date (YYYY-MM-DD)')
        self.label_end   = TextBox(self.label_end_ax, '', initial='End Date (YYYY-MM-DD)')
        for box in [self.label_stock, self.label_start, self.label_end]:
            box.set_active(False)

        # Input fields
        self.stock_box_ax = plt.axes([0.1, 0.27, 0.2, 0.05])
        self.start_box_ax = plt.axes([0.4, 0.27, 0.2, 0.05])
        self.end_box_ax   = plt.axes([0.7, 0.27, 0.2, 0.05])
        self.stock_box = TextBox(self.stock_box_ax, '', initial="AAPL")
        self.start_box = TextBox(self.start_box_ax, '', initial="2020-01-01")
        self.end_box   = TextBox(self.end_box_ax, '', initial="2023-01-01")

        # Visualize button
        self.button_ax = plt.axes([0.45, 0.15, 0.1, 0.06])
        self.visualize_button = Button(self.button_ax, 'Visualize')
        self.visualize_button.on_clicked(self.visualize)

        self.ax.set_title("Enter inputs and click Visualize")
        self.ax.grid(True)

        # Tooltip annotation
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(20, 20),
                                      textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        # State
        self.df = None
        self.line = None
        self.hover_text = None

        # Pan tracking
        self.is_panning = False
        self.pan_start = (0, 0)        # pixel position
        self.lim_start = ((), ())      # initial axis limits

        # Event bindings
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_pan)

    def fetch_data(self, stock, start, end):
        try:
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD.")
            return pd.DataFrame()

        df = yf.download(stock, start=start, end=end, auto_adjust=False)
        if df.empty:
            print("❌ No data found.")
        return df

    def visualize(self, event):
        self.ax.clear()
        self.df = None
        stock = self.stock_box.text.strip().upper()
        start = self.start_box.text.strip()
        end = self.end_box.text.strip()

        df = self.fetch_data(stock, start, end)
        if df.empty:
            self.ax.set_title("No data available for input.")
            self.fig.canvas.draw_idle()
            return

        self.df = df.copy()
        self.line, = self.ax.plot(df.index, df['Close'], label='Close Price', color='blue')
        self.ax.set_title(f'{stock} Closing Price from {start} to {end}')
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price (USD)")
        self.ax.legend()
        self.ax.grid(True)

        self.annot.set_visible(False)

        # Recreate hover text box
        self.hover_text = self.ax.text(
            0.99, 0.95, "", transform=self.ax.transAxes,
            ha='right', va='top', fontsize=13,
            bbox=dict(boxstyle="round", fc="yellow", ec="black", alpha=0.9)
        )
        self.hover_text.set_visible(False)

        self.fig.canvas.draw_idle()

    def on_hover(self, event):
        if event.inaxes == self.ax and self.df is not None and self.line is not None:
            xdata = self.df.index
            ydata = self.df['Close'].values
            x_float = mdates.date2num(xdata)

            if event.xdata is None or event.ydata is None:
                self.annot.set_visible(False)
                if self.hover_text:
                    self.hover_text.set_visible(False)
                self.fig.canvas.draw_idle()
                return

            ind = min(range(len(x_float)), key=lambda i: abs(x_float[i] - event.xdata))
            x = xdata[ind]
            val = ydata[ind]
            if isinstance(val, (list, tuple, np.ndarray)):
                val = val[0]
            y = float(val)

            if abs(x_float[ind] - event.xdata) < 1.5:
                self.annot.xy = (x, y)
                self.annot.set_text(f"{x.strftime('%Y-%m-%d')}\n${y:.2f}")
                self.annot.set_visible(True)
                if self.hover_text:
                    self.hover_text.set_text(f"{x.strftime('%Y-%m-%d')} | ${y:.2f}")
                    self.hover_text.set_visible(True)
            else:
                self.annot.set_visible(False)
                if self.hover_text:
                    self.hover_text.set_visible(False)

            self.fig.canvas.draw_idle()
        else:
            self.annot.set_visible(False)
            if self.hover_text:
                self.hover_text.set_visible(False)
            self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        base_scale = 1.1
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            return

        new_xlim = [
            xdata - (xdata - cur_xlim[0]) * scale_factor,
            xdata + (cur_xlim[1] - xdata) * scale_factor
        ]
        new_ylim = [
            ydata - (ydata - cur_ylim[0]) * scale_factor,
            ydata + (cur_ylim[1] - ydata) * scale_factor
        ]

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self.is_panning = True
            self.pan_start = (event.x, event.y)  # pixel coords
            self.lim_start = (self.ax.get_xlim(), self.ax.get_ylim())

    def on_release(self, event):
        if event.button == 1:
            self.is_panning = False

    def on_pan(self, event):
        if self.is_panning and event.inaxes == self.ax and event.x is not None and event.y is not None:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]

            xlim, ylim = self.lim_start
            bbox = self.ax.get_window_extent()
            width = bbox.width
            height = bbox.height

            dx_data = (xlim[1] - xlim[0]) * dx / width
            dy_data = (ylim[1] - ylim[0]) * dy / height

            self.ax.set_xlim(xlim[0] - dx_data, xlim[1] - dx_data)
            self.ax.set_ylim(ylim[0] - dy_data, ylim[1] - dy_data)
            self.fig.canvas.draw_idle()

def main():
    app = StockVisualizer()
    plt.show()

if __name__ == "__main__":
    main()
