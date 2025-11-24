import sys
import random
from math import inf
from typing import List, Tuple, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QAction, QToolBar, QPushButton,
    QDialog, QCheckBox, QSpinBox, QDialogButtonBox, QGroupBox
)
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap
from PyQt5.QtCore import (
    Qt, QSize, QRect, QPoint, QPropertyAnimation, QTimer
)

Player = int                    # 1 或 2
StackBoard = List[List[List[Player]]]

# ====================== 全局参数 ======================

BOARD_ROWS = 4
BOARD_COLS = 4
TOTAL_TILES_PER_PLAYER = 5
TOTAL_TILES = TOTAL_TILES_PER_PLAYER * 2

# AI 相关参数
USE_AI = True          # 人机对战：人类 = 1，AI = 2
AI_PLAYER = 2
AI_SEARCH_DEPTH = 12    # 搜索深度（调大更聪明也更慢）


# ====================== 规则配置 ======================

class GameRules:
    def __init__(self):
        self.anti_backtracking = True
        self.n_move_rule = True
        self.n_move_limit = 5
        self.stalemate_rule = True
        self.stalemate_limit = 20

class SettingsDialog(QDialog):
    def __init__(self, rules: GameRules, parent=None):
        super().__init__(parent)
        self.setWindowTitle("特殊规则设置")
        self.resize(420, 380)
        # 拷贝一份规则用于编辑，确认后再应用
        self.rules = GameRules()
        self.rules.anti_backtracking = rules.anti_backtracking
        self.rules.n_move_rule = rules.n_move_rule
        self.rules.n_move_limit = rules.n_move_limit
        self.rules.stalemate_rule = rules.stalemate_rule
        self.rules.stalemate_limit = rules.stalemate_limit
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 1. 反悔棋禁手
        self.group_anti = QGroupBox("反悔棋禁手 (Anti-Backtracking)")
        self.group_anti.setCheckable(True)
        self.group_anti.setChecked(self.rules.anti_backtracking)
        
        layout_anti = QVBoxLayout()
        lbl_anti = QLabel("禁止将刚移动过的棋子立刻移回原位，防止恶意循环。")
        lbl_anti.setStyleSheet("color: #666; font-size: 12px;")
        layout_anti.addWidget(lbl_anti)
        self.group_anti.setLayout(layout_anti)
        layout.addWidget(self.group_anti)

        # 2. N 连动判负
        self.group_n = QGroupBox("N 连动判负")
        self.group_n.setCheckable(True)
        self.group_n.setChecked(self.rules.n_move_rule)

        layout_n = QHBoxLayout()
        layout_n.addWidget(QLabel("最大允许连续行动次数:"))
        self.sb_n_move = QSpinBox()
        self.sb_n_move.setRange(3, 20)
        self.sb_n_move.setValue(self.rules.n_move_limit)
        layout_n.addWidget(self.sb_n_move)
        layout_n.addStretch()
        self.group_n.setLayout(layout_n)
        layout.addWidget(self.group_n)

        # 3. 无进展平局
        self.group_s = QGroupBox("无进展平局")
        self.group_s.setCheckable(True)
        self.group_s.setChecked(self.rules.stalemate_rule)

        layout_s = QHBoxLayout()
        layout_s.addWidget(QLabel("最大无消减换手次数:"))
        self.sb_stalemate = QSpinBox()
        self.sb_stalemate.setRange(10, 100)
        self.sb_stalemate.setValue(self.rules.stalemate_limit)
        layout_s.addWidget(self.sb_stalemate)
        layout_s.addStretch()
        self.group_s.setLayout(layout_s)
        layout.addWidget(self.group_s)

        layout.addStretch()

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_rules(self):
        self.rules.anti_backtracking = self.group_anti.isChecked()
        self.rules.n_move_rule = self.group_n.isChecked()
        self.rules.n_move_limit = self.sb_n_move.value()
        self.rules.stalemate_rule = self.group_s.isChecked()
        self.rules.stalemate_limit = self.sb_stalemate.value()
        return self.rules


# ====================== 基本游戏逻辑 ======================

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS


def get_nonempty_cells(board: StackBoard) -> List[Tuple[int, int]]:
    cells = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if board[r][c]:
                cells.append((r, c))
    return cells


def is_connected(board: StackBoard) -> bool:
    """检查所有非空格子是否四连通。"""
    cells = get_nonempty_cells(board)
    if len(cells) <= 1:
        return True
    from collections import deque
    start = cells[0]
    q = deque([start])
    visited = {start}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while q:
        r, c = q.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                continue
            if not board[nr][nc]:
                continue
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc))
    return len(visited) == len(cells)


def simulate_move(board: StackBoard,
                  src: Tuple[int, int],
                  dst: Tuple[int, int]) -> Optional[StackBoard]:
    """尝试从 src 移动顶牌到 dst；若不连通，则返回 None。"""
    r1, c1 = src
    r2, c2 = dst
    if not in_bounds(r1, c1) or not in_bounds(r2, c2):
        return None
    if not board[r1][c1]:
        return None
    if not board[r2][c2]:
        return None

    # 深拷贝
    new_board: StackBoard = [[stack.copy() for stack in row] for row in board]
    card = new_board[r1][c1].pop()
    new_board[r2][c2].append(card)

    if not is_connected(new_board):
        return None
    return new_board


def generate_legal_moves(board: StackBoard, player: Player, forbidden_move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """返回所有合法走法 (src, dst)。"""
    moves = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            stack = board[r][c]
            if not stack:
                continue
            if stack[-1] != player:
                continue
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    continue
                if not board[nr][nc]:
                    continue
                
                # 检查是否是禁止的走法（例如对方无棋可走时不能移回原位）
                if forbidden_move and ((r, c), (nr, nc)) == forbidden_move:
                    continue

                if simulate_move(board, (r, c), (nr, nc)) is not None:
                    moves.append(((r, c), (nr, nc)))
    return moves


def top_winner_if_single_stack(board: StackBoard) -> Optional[Player]:
    """若只剩一堆牌，则返回最顶牌所属玩家，否则返回 None。"""
    cells = get_nonempty_cells(board)
    if len(cells) != 1:
        return None
    r, c = cells[0]
    if not board[r][c]:
        return None
    return board[r][c][-1]


def check_bottom_control_winner(board: StackBoard) -> Optional[Player]:
    """如果所有非空堆的最底层牌都属于同一玩家，则该玩家获胜。"""
    cells = get_nonempty_cells(board)
    if not cells:
        return None
    
    first_r, first_c = cells[0]
    if not board[first_r][first_c]:
        return None
    
    potential_winner = board[first_r][first_c][0]
    
    for r, c in cells[1:]:
        if not board[r][c]:
            continue
        if board[r][c][0] != potential_winner:
            return None
            
    return potential_winner


def find_highest_stack_winner(board: StackBoard) -> Player:
    """双方都无棋可走但还不止一堆时：找最高牌堆的顶牌所属玩家。"""
    cells = get_nonempty_cells(board)
    assert cells, "理论上不应出现完全空棋盘"
    best_r, best_c = cells[0]
    best_h = len(board[best_r][best_c])
    for r, c in cells[1:]:
        h = len(board[r][c])
        if h > best_h:
            best_h = h
            best_r, best_c = r, c
    return board[best_r][best_c][-1]


def generate_random_initial_board() -> StackBoard:
    """在 6x6 棋盘上随机生成一个 10 格连通区域，并在其中随机放置 5 个 1、5 个 2。"""
    cells = set()
    start_r = random.randrange(BOARD_ROWS)
    start_c = random.randrange(BOARD_COLS)
    cells.add((start_r, start_c))
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while len(cells) < TOTAL_TILES:
        r, c = random.choice(list(cells))
        dr, dc = random.choice(dirs)
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc):
            continue
        cells.add((nr, nc))

    board: StackBoard = [[[] for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

    cells_list = list(cells)
    random.shuffle(cells_list)
    p1_cells = cells_list[:TOTAL_TILES_PER_PLAYER]
    p2_cells = cells_list[TOTAL_TILES_PER_PLAYER:TOTAL_TILES]

    for r, c in p1_cells:
        board[r][c].append(1)
    for r, c in p2_cells:
        board[r][c].append(2)

    assert is_connected(board)
    return board


# ====================== AI 评价函数 & 搜索 ======================

def evaluate_board(board: StackBoard, ai_player: Player) -> float:
    """
    简单启发式：
    - 每堆：顶牌是 ai -> +高度^2；顶牌是对手 -> -高度^2
    - 最高堆：额外 +/- 5 * 高度
    - 总牌数差：*(0.5) 小修正
    """
    score = 0.0
    cells = get_nonempty_cells(board)
    max_h = 0
    max_owner = None

    total_ai = 0
    total_op = 0
    for r, c in cells:
        stack = board[r][c]
        h = len(stack)
        top = stack[-1]
        if top == ai_player:
            score += h * h
        else:
            score -= h * h
        if h > max_h:
            max_h = h
            max_owner = top
        for p in stack:
            if p == ai_player:
                total_ai += 1
            else:
                total_op += 1

    if max_owner == ai_player:
        score += 5 * max_h
    elif max_owner is not None:
        score -= 5 * max_h

    score += (total_ai - total_op) * 0.5
    return score


def alpha_beta(board: StackBoard,
               current_player: Player,
               skipped_last: bool,
               consecutive_moves: int,
               last_mover: Optional[Player],
               last_move_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
               switches_without_reduction: int,
               rules: GameRules,
               depth: int,
               alpha: float,
               beta: float,
               ai_player: Player) -> float:
    """极大极小 + alpha-beta 剪枝。"""
    # 终局 1：已经收缩成一堆
    winner_single = top_winner_if_single_stack(board)
    if winner_single is not None:
        if winner_single == ai_player:
            return 10000.0
        else:
            return -10000.0

    # 终局 1.5：底层控制获胜
    winner_bottom = check_bottom_control_winner(board)
    if winner_bottom is not None:
        if winner_bottom == ai_player:
            return 10000.0
        else:
            return -10000.0
            
    # 终局 1.8：平局判定
    if rules.stalemate_rule and switches_without_reduction >= rules.stalemate_limit:
        return 0.0

    # 计算禁止走法：如果上一手跳过（即对方无棋可走），则不能移回原位
    forbidden_move = None
    if rules.anti_backtracking and skipped_last and last_move_coords:
        src, dst = last_move_coords
        forbidden_move = (dst, src)

    legal_moves = generate_legal_moves(board, current_player, forbidden_move)

    # 终局 2：当前无棋可走且上一手也无棋可走
    if not legal_moves and skipped_last:
        winner2 = find_highest_stack_winner(board)
        if winner2 == ai_player:
            return 9000.0
        else:
            return -9000.0

    # 深度用完
    if depth == 0:
        return evaluate_board(board, ai_player)

    # 当前无棋可走，但上一手没跳过 -> pass 一次
    if not legal_moves:
        return alpha_beta(board,
                          3 - current_player,
                          True,
                          consecutive_moves,
                          last_mover,
                          last_move_coords,  # 传递上一步的移动坐标，因为这一步没动
                          switches_without_reduction, # pass 不改变堆数也不算换手（或者算？规则说换手了20次，pass导致换手）
                          rules,
                          depth - 1,
                          alpha,
                          beta,
                          ai_player)

    # 正常有棋可走
    current_stack_count = len(get_nonempty_cells(board))
    
    if current_player == ai_player:
        # 极大
        value = -inf
        for (src, dst) in legal_moves:
            # 计算新的连续移动次数
            new_consecutive = consecutive_moves
            if last_mover == current_player:
                new_consecutive += 1
            else:
                new_consecutive = 1
            
            # 检查连续移动判负规则
            if rules.n_move_rule and new_consecutive >= rules.n_move_limit:
                # AI 判负
                child_val = -20000.0
            else:
                new_board = simulate_move(board, src, dst)
                if new_board is None:
                    continue
                
                # 更新平局计数
                new_stack_count = len(get_nonempty_cells(new_board))
                new_switches = switches_without_reduction
                if new_stack_count < current_stack_count:
                    new_switches = 0
                elif new_consecutive == 1: # 发生了换手
                    new_switches += 1
                
                child_val = alpha_beta(new_board,
                                       3 - current_player,
                                       False,
                                       new_consecutive,
                                       current_player,
                                       (src, dst),  # 更新最后一步移动坐标
                                       new_switches,
                                       rules,
                                       depth - 1,
                                       alpha,
                                       beta,
                                       ai_player)
            
            if child_val > value:
                value = child_val
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break
        return value
    else:
        # 极小（对手）
        value = inf
        for (src, dst) in legal_moves:
            # 计算新的连续移动次数
            new_consecutive = consecutive_moves
            if last_mover == current_player:
                new_consecutive += 1
            else:
                new_consecutive = 1
            
            # 检查连续移动判负规则
            if rules.n_move_rule and new_consecutive >= rules.n_move_limit:
                # 对手判负 -> AI 获胜
                child_val = 20000.0
            else:
                new_board = simulate_move(board, src, dst)
                if new_board is None:
                    continue
                
                # 更新平局计数
                new_stack_count = len(get_nonempty_cells(new_board))
                new_switches = switches_without_reduction
                if new_stack_count < current_stack_count:
                    new_switches = 0
                elif new_consecutive == 1: # 发生了换手
                    new_switches += 1

                child_val = alpha_beta(new_board,
                                       3 - current_player,
                                       False,
                                       new_consecutive,
                                       current_player,
                                       (src, dst),  # 更新最后一步移动坐标
                                       new_switches,
                                       rules,
                                       depth - 1,
                                       alpha,
                                       beta,
                                       ai_player)
            
            if child_val < value:
                value = child_val
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        return value


def choose_ai_move(board: StackBoard,
                   current_player: Player,
                   skipped_last: bool,
                   consecutive_moves: int,
                   last_mover: Optional[Player],
                   forbidden_move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
                   switches_without_reduction: int,
                   rules: GameRules,
                   ai_player: Player,
                   depth: int) -> Tuple[Optional[Tuple[Tuple[int, int], Tuple[int, int]]], float]:
    """为 AI 选择一步走法，并返回评估分数。"""
    legal_moves = generate_legal_moves(board, current_player, forbidden_move)
    if not legal_moves:
        return None, 0.0

    best_val = -inf
    best_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    
    current_stack_count = len(get_nonempty_cells(board))

    for (src, dst) in legal_moves:
        # 计算新的连续移动次数
        new_consecutive = consecutive_moves
        if last_mover == current_player:
            new_consecutive += 1
        else:
            new_consecutive = 1
        
        # 检查连续移动判负规则
        if rules.n_move_rule and new_consecutive >= rules.n_move_limit:
            # 这一步会导致 AI 判负，极差
            val = -20000.0
        else:
            new_board = simulate_move(board, src, dst)
            if new_board is None:
                continue
            
            # 更新平局计数
            new_stack_count = len(get_nonempty_cells(new_board))
            new_switches = switches_without_reduction
            if new_stack_count < current_stack_count:
                new_switches = 0
            elif new_consecutive == 1: # 发生了换手
                new_switches += 1
            
            val = alpha_beta(new_board,
                             3 - current_player,
                             False,
                             new_consecutive,
                             current_player,
                             (src, dst),  # 传递当前移动作为 last_move_coords
                             new_switches,
                             rules,
                             depth - 1,
                             -inf,
                             inf,
                             ai_player)
        
        if val > best_val + 1e-9:
            best_val = val
            best_moves = [(src, dst)]
        elif abs(val - best_val) <= 1e-9:
            best_moves.append((src, dst))

    if not best_moves:
        # 如果所有走法都导致输（例如都被迫连续移动第5次），随机选一个
        return random.choice(legal_moves), best_val
    return random.choice(best_moves), best_val


# ====================== 自绘格子（伪 3D 叠牌） ======================

class StackWidget(QWidget):
    """棋盘上的一个格子，用自绘实现伪 3D 牌堆。"""

    def __init__(self, row: int, col: int, window: "ToppenWindow"):
        super().__init__(window)
        self.row = row
        self.col = col
        self.window = window
        self.setMinimumSize(QSize(100, 100))
        self.setMaximumSize(QSize(120, 120))
        self.setAttribute(Qt.WA_StyledBackground, True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.window.cell_clicked(self.row, self.col)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        board = self.window.board
        stack = board[self.row][self.col]

        # 空格子：什么都不画（“不显示”）
        if not stack:
            return

        w = self.width()
        h = self.height()
        n = len(stack)

        # 设定单张牌的大小（留出足够空间给堆叠偏移）
        # 之前是 w - 24，现在稍微缩小一点以防堆太高溢出，或者动态计算
        # 这里先给一个基础大小，然后根据堆叠数量动态调整偏移量
        base_margin = 14
        tile_w = w - 2 * base_margin
        tile_h = h - 2 * base_margin

        # 默认偏移量
        step_x = 4.0
        step_y = 4.0

        # 如果堆叠太高，压缩偏移量以适应控件大小
        # 堆叠总占用宽度 = tile_w + (n-1) * step_x
        # 我们希望 total_w <= w - 4 (留点边)
        if n > 1:
            max_spread_x = w - tile_w - 4
            max_spread_y = h - tile_h - 4
            if (n - 1) * step_x > max_spread_x:
                step_x = max_spread_x / (n - 1)
            if (n - 1) * step_y > max_spread_y:
                step_y = max_spread_y / (n - 1)

        # 计算整个堆叠的包围盒大小
        stack_total_w = tile_w + (n - 1) * step_x
        stack_total_h = tile_h + (n - 1) * step_y

        # 居中起始位置 (最底层牌的左上角)
        # 视觉上：底层牌在左上，顶层牌在右下
        start_x = (w - stack_total_w) / 2
        start_y = (h - stack_total_h) / 2

        selected = (self.window.selected_cell == (self.row, self.col))

        for index, player in enumerate(stack):
            # 从底到顶画 (index 0 是底)
            # 偏移方向：向右下 (dx, dy)
            cur_x = start_x + index * step_x
            cur_y = start_y + index * step_y
            
            # 使用 QRectF 以支持浮点坐标
            rect = QRect(int(cur_x), int(cur_y), int(tile_w), int(tile_h))

            # 颜色：玩家 1 蓝，玩家 2 红，下面几层略微调暗
            # 顶层是 index = n-1
            # 倒数第 k 层： offset = n - 1 - index
            offset = n - 1 - index
            
            if player == 1:
                base_color = QColor(180, 210, 255)
                border_color = QColor(20, 40, 90)
            else:
                base_color = QColor(255, 200, 200)
                border_color = QColor(120, 30, 30)

            factor = 1.0 - 0.08 * offset
            factor = max(0.6, factor)
            color = QColor(
                int(base_color.red() * factor),
                int(base_color.green() * factor),
                int(base_color.blue() * factor),
            )

            # 轻微阴影
            shadow_rect = rect.translated(2, 2)
            painter.setBrush(QColor(0, 0, 0, 60))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(shadow_rect, 6, 6)

            # 本体
            painter.setBrush(color)
            painter.setPen(border_color)
            painter.drawRoundedRect(rect, 6, 6)

            # 绘制图标 (空心)
            icon_color = QColor(0, 30, 80) if player == 1 else QColor(80, 10, 10)
            painter.setBrush(Qt.NoBrush)
            pen = QPen(icon_color)
            pen.setWidth(2)
            painter.setPen(pen)
            
            cx, cy = rect.center().x(), rect.center().y()
            w_icon = rect.width()
            h_icon = rect.height()
            
            if player == 1: # Human (Player 1)
                # Head
                head_r = int(w_icon * 0.14)
                painter.drawEllipse(QPoint(cx, int(cy - h_icon * 0.15)), head_r, head_r)
                # Body (Shoulders)
                body_w = int(w_icon * 0.45)
                body_h = int(h_icon * 0.3)
                # Draw chord for shoulders (0 to 180 degrees is top half)
                painter.drawChord(int(cx - body_w/2), int(cy - h_icon * 0.05), body_w, body_h * 2, 0, 180 * 16)
            else: # Computer (Player 2)
                # Screen
                screen_w = int(w_icon * 0.45)
                screen_h = int(h_icon * 0.32)
                painter.drawRoundedRect(int(cx - screen_w/2), int(cy - h_icon * 0.2), screen_w, screen_h, 3, 3)
                # Base/Stand
                base_w = int(w_icon * 0.25)
                base_h = int(h_icon * 0.06)
                painter.drawRect(int(cx - base_w/2), int(cy + h_icon * 0.18), base_w, base_h)
                # Neck
                neck_w = int(w_icon * 0.1)
                neck_h = int(h_icon * 0.1)
                painter.drawRect(int(cx - neck_w/2), int(cy + h_icon * 0.12), neck_w, neck_h)

            # 如果是顶层牌且被选中，画选中框
            if selected and index == n - 1:
                painter.setPen(QColor("#ff9900"))
                painter.setBrush(Qt.NoBrush)
                # 稍微外扩一点
                outline_rect = rect.adjusted(-2, -2, 2, 2)
                painter.drawRoundedRect(outline_rect, 8, 8)


# ====================== 主窗口（带动画） ======================

class ToppenWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("登山家 Toppen")

        self.board: StackBoard = [[[] for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.current_player: Player = 1
        self.selected_cell: Optional[Tuple[int, int]] = None
        self.skipped_last_turn: bool = False
        self.game_over: bool = False

        self.consecutive_moves: int = 0
        self.last_moving_player: Optional[Player] = None
        self.last_move_record: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.switches_without_reduction: int = 0
        # 当对方被迫跳过时，禁止将刚走过的棋子移回原位
        self.forbidden_move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        
        self.rules = GameRules()

        # AI 设置
        self.use_ai = USE_AI
        self.ai_player = AI_PLAYER

        # 动画相关
        self.animating: bool = False
        self.animation_widget: Optional[QWidget] = None
        self.animation: Optional[QPropertyAnimation] = None
        self.pending_board_after_move: Optional[StackBoard] = None
        self.anim_next_player: Optional[Player] = None

        self.init_ui()
        self.new_game()

    def init_ui(self):
        central = QWidget()
        central.setStyleSheet("background-color: white;")
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 顶部信息栏：显示 AI 评估状态
        top_layout = QHBoxLayout()
        top_layout.addStretch()
        
        self.indicator = QLabel()
        self.indicator.setFixedSize(20, 20)
        # 默认灰色
        self.indicator.setStyleSheet("background-color: gray; border-radius: 10px; border: 1px solid #666;")
        self.indicator.setToolTip("AI 评估：未知")
        top_layout.addWidget(self.indicator)
        
        main_layout.addLayout(top_layout)

        # 棋盘布局
        self.grid = QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(4)
        self.grid.setVerticalSpacing(4)
        main_layout.addLayout(self.grid)

        self.cells: List[List[StackWidget]] = []
        for r in range(BOARD_ROWS):
            row_widgets = []
            for c in range(BOARD_COLS):
                cell = StackWidget(r, c, self)
                self.grid.addWidget(cell, r, c)
                row_widgets.append(cell)
            self.cells.append(row_widgets)

        self.setCentralWidget(central)

        # 工具栏：新游戏
        toolbar = QToolBar("MainToolbar")
        self.addToolBar(toolbar)
        
        new_game_btn = QPushButton("新游戏")
        new_game_btn.setStyleSheet("font-size: 14px; padding: 6px 12px; font-weight: bold;")
        new_game_btn.clicked.connect(self.new_game)
        toolbar.addWidget(new_game_btn)
        
        rules_btn = QPushButton("特殊规则")
        rules_btn.setStyleSheet("font-size: 14px; padding: 6px 12px; font-weight: bold;")
        rules_btn.clicked.connect(self.open_rules_dialog)
        toolbar.addWidget(rules_btn)

        self.resize(550, 600)

    # ---------- 游戏控制 ----------

    def open_rules_dialog(self):
        dialog = SettingsDialog(self.rules, self)
        if dialog.exec_() == QDialog.Accepted:
            self.rules = dialog.get_rules()
            self.new_game()

    def new_game(self):
        if self.animation is not None:
            self.animation.stop()
            self.animation = None
        if self.animation_widget is not None:
            self.animation_widget.hide()
            self.animation_widget.deleteLater()
            self.animation_widget = None

        self.board = generate_random_initial_board()
        self.current_player = 1
        self.selected_cell = None
        self.skipped_last_turn = False
        self.game_over = False
        self.consecutive_moves = 0
        self.last_moving_player = None
        self.last_move_record = None
        self.switches_without_reduction = 0
        self.animating = False
        self.pending_board_after_move = None
        self.anim_next_player = None
        self.forbidden_move = None
        # 重置指示器
        self.indicator.setStyleSheet("background-color: gray; border-radius: 10px; border: 1px solid #666;")
        self.indicator.setToolTip("AI 评估：未知")
        
        self.update_board_view()
        self.start_turn()

    def start_turn(self):
        if self.game_over or self.animating:
            return

        # 终局 1：收缩成一堆
        winner = top_winner_if_single_stack(self.board)
        if winner is not None:
            self.game_over = True
            self.update_board_view()
            name = "您" if winner == 1 else "电脑"
            QMessageBox.information(self, "游戏结束",
                                    f"{name}获胜。")
            # self.status_label.setText(f"游戏结束：玩家 {winner} 获胜。")
            self.new_game()
            return

        # 终局 1.5：底层控制获胜
        winner_bottom = check_bottom_control_winner(self.board)
        if winner_bottom is not None:
            self.game_over = True
            self.update_board_view()
            name = "您" if winner_bottom == 1 else "电脑"
            QMessageBox.information(self, "游戏结束",
                                    f"{name}获胜（控制了所有底层）。")
            self.new_game()
            return

        # 计算禁止走法：如果上一手跳过（即对方无棋可走），则不能移回原位
        if self.rules.anti_backtracking and self.skipped_last_turn and self.last_move_record:
            src, dst = self.last_move_record
            self.forbidden_move = (dst, src)
        else:
            self.forbidden_move = None

        legal_moves = generate_legal_moves(self.board, self.current_player, self.forbidden_move)
        if not legal_moves:
            # 无棋可走
            if self.skipped_last_turn:
                # 连续两方都无棋可走
                self.game_over = True
                winner2 = find_highest_stack_winner(self.board)
                self.update_board_view()
                QMessageBox.information(self, "游戏结束",
                                        f"平局：双方都没有合法走法。")
                # self.status_label.setText(f"游戏结束：玩家 {winner2} 获胜。")
                self.new_game()
                return
            else:
                # 当前方跳过
                # 用状态栏提示替代弹窗
                if self.use_ai and self.current_player == self.ai_player:
                    pass
                    # self.status_label.setText(f"电脑（玩家 {self.current_player}）无棋可走，本回合跳过。")
                else:
                    pass
                    # self.status_label.setText(f"玩家 {self.current_player} 没有任何合法走法，本回合跳过。")
                self.skipped_last_turn = True
                self.current_player = 2 if self.current_player == 1 else 1
                self.selected_cell = None
                self.update_board_view()
                self.start_turn()
                return

        # 有棋可走
        self.skipped_last_turn = False
        self.selected_cell = None
        self.update_board_view()

        if self.use_ai and self.current_player == self.ai_player:
            # 电脑回合，延迟执行以让 UI 刷新
            QTimer.singleShot(50, lambda: self.run_ai_turn(self.forbidden_move))
        else:
            # 人类回合
            pass
            # self.status_label.setText(
            #     f"轮到玩家 {self.current_player}："
            #     f"先点击自己的顶牌格子，再点击相邻有牌格子。"
            # )

    def run_ai_turn(self, forbidden_move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]):
        if self.game_over:
            return

        move, score = choose_ai_move(self.board,
                              self.current_player,
                              self.skipped_last_turn,
                              self.consecutive_moves,
                              self.last_moving_player,
                              forbidden_move,
                              self.switches_without_reduction,
                              self.rules,
                              self.ai_player,
                              AI_SEARCH_DEPTH)
        
        self.update_indicator(score)

        if move is None:
            self.skipped_last_turn = True
            self.current_player = 2 if self.current_player == 1 else 1
            self.start_turn()
            return

        src, dst = move
        self.animate_move(src, dst, next_player=2 if self.current_player == 1 else 1)

    def update_indicator(self, score: float):
        """根据 AI 评分更新指示器颜色。"""
        # 阈值设定：超过 5000 认为是必胜/必败
        WIN_THRESHOLD = 5000.0
        
        if score > WIN_THRESHOLD:
            # AI (Player 2) 必胜 -> 红色
            color = "red"
            tooltip = "电脑必胜"
        elif score < -WIN_THRESHOLD:
            # Human (Player 1) 必胜 -> 绿色
            color = "#00cc00"
            tooltip = "人类必胜"
        elif abs(score) < 0.1:
            # 平局 -> 蓝色
            color = "blue"
            tooltip = "预计平局"
        else:
            # 局势不明 -> 灰色
            color = "gray"
            tooltip = "局势不明"
            
        self.indicator.setStyleSheet(f"background-color: {color}; border-radius: 10px; border: 1px solid #666;")
        self.indicator.setToolTip(tooltip)

    def cell_clicked(self, r: int, c: int):
        if self.game_over or self.animating:
            return
        if self.use_ai and self.current_player == self.ai_player:
            # AI 回合，禁点
            return

        stack = self.board[r][c]

        # 第一次点击：选起点
        if self.selected_cell is None:
            if not stack:
                return
            if stack[-1] != self.current_player:
                # 用闪烁提示替代弹窗（在被点击的格子上闪烁）
                self.flash_cell(r, c)
                return
            self.selected_cell = (r, c)
            self.update_board_view()
            return

        # 已经选过起点
        sr, sc = self.selected_cell

        # 再点起点 -> 取消选择
        if (r, c) == (sr, sc):
            self.selected_cell = None
            self.update_board_view()
            return

        # 第二步：尝试移动到 (r, c)
        if abs(sr - r) + abs(sc - c) != 1:
            self.flash_cell(sr, sc)
            self.selected_cell = None
            self.update_board_view()
            return

        if not self.board[r][c]:
            self.flash_cell(sr, sc)
            self.selected_cell = None
            self.update_board_view()
            return

        if not self.board[sr][sc] or self.board[sr][sc][-1] != self.current_player:
            self.flash_cell(sr, sc)
            self.selected_cell = None
            self.update_board_view()
            return

        # 检查是否是禁止的走法（例如对方无棋可走时不能移回原位）
        if self.forbidden_move and ((sr, sc), (r, c)) == self.forbidden_move:
            self.flash_cell(sr, sc)
            self.selected_cell = None
            self.update_board_view()
            return

        # 合法走法 -> 动画移动
        self.skipped_last_turn = False
        self.animate_move((sr, sc), (r, c),
                          next_player=2 if self.current_player == 1 else 1)

    # ---------- 动画 ----------

    def flash_cell(self, r: int, c: int, flashes: int = 2, interval: int = 220):
        """在格子上创建半透明红色覆盖并闪烁指定次数（用于代替弹窗提示）。"""
        try:
            cell_widget = self.cells[r][c]
        except Exception:
            return

        # 创建覆盖层
        overlay = QLabel(cell_widget)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        overlay.setStyleSheet("background-color: rgba(255,0,0,120); border-radius: 8px;")
        overlay.setGeometry(cell_widget.rect())
        overlay.setVisible(False)
        overlay.show()

        # 计时器在 overlay 上，以便生命周期与之绑定
        timer = QTimer(overlay)
        total = flashes * 2

        def _tick():
            nonlocal total
            overlay.setVisible(not overlay.isVisible())
            total -= 1
            if total <= 0:
                timer.stop()
                overlay.hide()
                overlay.deleteLater()

        timer.timeout.connect(_tick)
        timer.start(interval)

    def animate_move(self, src: Tuple[int, int], dst: Tuple[int, int], next_player: Player):
        """做一次带动画的移动：从 src 到 dst。"""
        if self.animating:
            return

        # 记录移动前的堆数
        old_stack_count = len(get_nonempty_cells(self.board))

        if self.current_player == self.last_moving_player:
            self.consecutive_moves += 1
        else:
            self.last_moving_player = self.current_player
            self.consecutive_moves = 1

        self.last_move_record = (src, dst)

        sr, sc = src
        new_board = simulate_move(self.board, src, dst)
        if new_board is None:
            # 对非法导致断连的走法，闪烁起点替代弹窗提示
            self.flash_cell(sr, sc)
            self.selected_cell = None
            self.update_board_view()
            return

        # 更新无消减换手计数
        new_stack_count = len(get_nonempty_cells(new_board))
        if new_stack_count < old_stack_count:
            self.switches_without_reduction = 0
        elif self.consecutive_moves == 1: # 发生了换手
            self.switches_without_reduction += 1

        self.animating = True
        self.anim_next_player = next_player
        self.pending_board_after_move = new_board

        sr, sc = src
        dr, dc = dst

        # 移动的那张牌是谁
        old_stack = self.board[sr][sc]
        if not old_stack:
            moving_player = None
        else:
            moving_player = old_stack[-1]

        # 显示用临时棋盘：只把 src 堆顶弹出，dst 先不加
        temp_board = [[stack.copy() for stack in row] for row in self.board]
        if temp_board[sr][sc]:
            temp_board[sr][sc].pop()
        self.board = temp_board
        self.selected_cell = None
        self.update_board_view()

        # 创建一个“飞行中的牌”小方块
        parent = self.centralWidget()
        src_widget = self.cells[sr][sc]
        dst_widget = self.cells[dr][dc]

        src_top_left = src_widget.mapTo(parent, QPoint(0, 0))
        dst_top_left = dst_widget.mapTo(parent, QPoint(0, 0))

        cell_w = src_widget.width()
        cell_h = src_widget.height()
        margin = 14
        tile_w = cell_w - 2 * margin
        tile_h = cell_h - 2 * margin

        start_rect = QRect(src_top_left.x() + margin,
                           src_top_left.y() + margin,
                           tile_w, tile_h)
        end_rect = QRect(dst_top_left.x() + margin,
                         dst_top_left.y() + margin,
                         tile_w, tile_h)

        # 绘制 QPixmap
        pixmap = QPixmap(tile_w, tile_h)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = QRect(0, 0, tile_w, tile_h)
        
        if moving_player == 1:
            base_color = QColor(180, 210, 255)
            border_color = QColor(20, 40, 90)
        else:
            base_color = QColor(255, 200, 200)
            border_color = QColor(120, 30, 30)
            
        painter.setBrush(base_color)
        painter.setPen(border_color)
        painter.drawRoundedRect(rect, 6, 6)
        
        # 绘制图标
        icon_color = QColor(0, 30, 80) if moving_player == 1 else QColor(80, 10, 10)
        painter.setBrush(Qt.NoBrush)
        pen = QPen(icon_color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        cx, cy = tile_w // 2, tile_h // 2
        w_icon = tile_w
        h_icon = tile_h
        
        if moving_player == 1: # Human
            head_r = int(w_icon * 0.14)
            painter.drawEllipse(QPoint(cx, int(cy - h_icon * 0.15)), head_r, head_r)
            body_w = int(w_icon * 0.45)
            body_h = int(h_icon * 0.3)
            painter.drawChord(int(cx - body_w/2), int(cy - h_icon * 0.05), body_w, body_h * 2, 0, 180 * 16)
        else: # Computer
            screen_w = int(w_icon * 0.45)
            screen_h = int(h_icon * 0.32)
            painter.drawRoundedRect(int(cx - screen_w/2), int(cy - h_icon * 0.2), screen_w, screen_h, 3, 3)
            base_w = int(w_icon * 0.25)
            base_h = int(h_icon * 0.06)
            painter.drawRect(int(cx - base_w/2), int(cy + h_icon * 0.18), base_w, base_h)
            neck_w = int(w_icon * 0.1)
            neck_h = int(h_icon * 0.1)
            painter.drawRect(int(cx - neck_w/2), int(cy + h_icon * 0.12), neck_w, neck_h)
            
        painter.end()

        # 用 QLabel 显示飞行牌
        tile_label = QLabel(parent)
        tile_label.setPixmap(pixmap)
        tile_label.setGeometry(start_rect)
        tile_label.show()
        tile_label.raise_()

        self.animation_widget = tile_label
        self.animation = QPropertyAnimation(tile_label, b"geometry")
        self.animation.setDuration(250)
        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.finished.connect(self.on_animation_finished)
        self.animation.start()

    def on_animation_finished(self):
        # 清理动画控件
        if self.animation_widget is not None:
            self.animation_widget.hide()
            self.animation_widget.deleteLater()
            self.animation_widget = None
        self.animation = None

        # 落子：切换到真实新棋盘
        if self.pending_board_after_move is not None:
            self.board = self.pending_board_after_move
            self.pending_board_after_move = None
        # 任何一次真实落子后，清除禁止回退的记号
        self.forbidden_move = None

        self.animating = False
        self.update_board_view()

        if self.rules.n_move_rule and self.consecutive_moves >= self.rules.n_move_limit:
            self.game_over = True
            loser = self.last_moving_player
            winner = 2 if loser == 1 else 1
            l_name = "您" if loser == 1 else "电脑"
            w_name = "您" if winner == 1 else "电脑"
            QMessageBox.information(self, "游戏结束",
                                    f"{l_name}违规：连续移动了 {self.rules.n_move_limit} 次。\n"
                                    f"{w_name}获胜。")
            self.new_game()
            return

        if self.rules.stalemate_rule and self.switches_without_reduction >= self.rules.stalemate_limit:
            self.game_over = True
            QMessageBox.information(self, "游戏结束",
                                    f"平局：连续 {self.rules.stalemate_limit} 次换手未减少牌堆数量。")
            self.new_game()
            return

        # 轮到下一位玩家
        if self.anim_next_player is not None:
            self.current_player = self.anim_next_player
            self.anim_next_player = None

        self.start_turn()

    # ---------- 视图刷新 ----------

    def update_board_view(self):
        for row_widgets in self.cells:
            for cell in row_widgets:
                cell.update()


def main():
    app = QApplication(sys.argv)
    window = ToppenWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()