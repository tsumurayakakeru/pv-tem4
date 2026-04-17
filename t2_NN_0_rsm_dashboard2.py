import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math
import csv
import sys

# --- modeFRONTIER書き出しクラス (最新版に適合) ---
class t2_NN_0:
    def __init__(self):
        self.n_input = 2
        try:
            # UTF-8-SIGで読み込み（BOM対策）
            with open('t2_NN_0.csv', encoding='utf-8-sig') as csvfile:
                filereader = csv.reader(csvfile)
                next(filereader)
                next(filereader)
                self.x_range = [[0 for _ in range(2)] for _ in range(2)]
                for i in range(2):
                    self.x_range[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.y_range = [0 for _ in range(2)]
                for i in range(2):
                    self.y_range[i] = float(next(filereader)[0])
                next(filereader)
                self.out_range = [0 for _ in range(2)]
                for i in range(2):
                    self.out_range[i] = float(next(filereader)[0])
                next(filereader)
                self.w1 = [[0 for _ in range(2)] for _ in range(15)]
                for i in range(15):
                    self.w1[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b1 = [0 for _ in range(15)]
                for i in range(15):
                    self.b1[i] = float(next(filereader)[0])
                next(filereader)
                self.w2 = [[0 for _ in range(15)] for _ in range(1)]
                for i in range(1):
                    self.w2[i] = [float(value) for value in next(filereader)]
                next(filereader)
                self.b2 = [0 for _ in range(1)]
                for i in range(1):
                    self.b2[i] = float(next(filereader)[0])
                csvfile.close()
        except Exception as e:
            raise Exception(f"CSV読み込みエラー: {e}")

    def evaluate(self, x):
        if len(x) != 5: # 入力は5個(R1, R2, R3, R4, R5)
            return math.nan
        xx = [x[1], x[3]] # R2 と R4 のみ使用
        xn = [0 for _ in range(self.n_input)]
        for i in range(self.n_input):
            xn[i] = (2 * xx[i] - self.x_range[i][0] - self.x_range[i][1]) / (self.x_range[i][1] - self.x_range[i][0])
        n1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            n1[i] = self.b1[i]
            for j in range(len(self.w1[0])):
                n1[i] += self.w1[i][j] * xn[j]
        y1 = [0 for _ in range(len(self.w1))]
        for i in range(len(self.w1)):
            try:
                exp = math.exp(-2.0 * n1[i])
                y1[i] = (1.0 - exp)/(1.0 + exp)
            except OverflowError:
                y1[i] = -1.0 if n1[i] > 0 else 1.0
        n2 = [0 for _ in range(len(self.w2))]
        for i in range(len(self.w2)):
            n2[i] = self.b2[i]
            for j in range(len(self.w2[0])):
                n2[i] += self.w2[i][j] * y1[j]
        yn = [n2[0]]
        y = self.y_range[0] + (self.y_range[1] - self.y_range[0])/(self.out_range[1] - self.out_range[0]) * (yn[0] - self.out_range[0])
        return y

# --- Streamlit ダッシュボード部分 ---
def main():
    st.set_page_config(layout="wide", page_title="modeFRONTIER Dashboard")
    st.title("📊 modeFRONTIER RSM Dashboard")

    try:
        model = t2_NN_0()
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}")
        st.info("t2_NN_0.csv がこのPythonファイルと同じフォルダにあることを確認してください。")
        return

    # サイドバー：スライダーの設定（範囲は自動取得したx_rangeを参考に設定すると良いです）
    st.sidebar.header("Input Parameters")
    
    # 実際の設計範囲が不明なため、暫定的に 0.0〜1.0 としています。適宜書き換えてください。
    r1 = st.sidebar.number_input("R1 (Ignored)", value=0.04)
    r2 = st.sidebar.slider("R2 (Active)", 0.01, 0.1, 0.05) 
    r3 = st.sidebar.number_input("R3 (Ignored)", value=0.0337)
    r4 = st.sidebar.slider("R4 (Active)", 1.0, 7.0, 4.0)
    r5 = st.sidebar.number_input("R5 (Ignored)", value=0.11)

    # 予測実行
    input_vec = [r1, r2, r3, r4, r5]
    t2_val = model.evaluate(input_vec)

    # 結果表示
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Predicted Output (t2)", value=f"{t2_val:.6f}")
        st.write("---")
        st.caption("Active Inputs: R2, R4")
        st.caption("Ignored: R1, R3, R5")

    # 3Dグラフ表示
    with col2:
        res = 25
        # スライダーと同じ範囲でグラフを描画
        r2_grid = np.linspace(0.01, 0.1, res)
        r4_grid = np.linspace(1.0, 7.0, res)
        R2, R4 = np.meshgrid(r2_grid, r4_grid)
        
        Z = np.zeros((res, res))
        for i in range(res):
            for j in range(res):
                Z[i, j] = model.evaluate([r1, R2[i, j], r3, R4[i, j], r5])

        fig = go.Figure(data=[go.Surface(z=Z, x=R2, y=R4, colorscale='Viridis')])
        fig.update_layout(
            scene=dict(xaxis_title='R2', yaxis_title='R4', zaxis_title='t2'),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    main()
