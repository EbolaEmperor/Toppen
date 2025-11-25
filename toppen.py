import sys
import random
import json
import os
import time
from math import inf
from typing import List, Tuple, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout,
    QVBoxLayout, QHBoxLayout, QLabel, QMessageBox, QAction, QPushButton,
    QDialog, QCheckBox, QSpinBox, QDialogButtonBox, QGroupBox, QFrame, QScrollArea,
    QInputDialog
)
from PyQt5.QtGui import QPainter, QColor, QPen, QPixmap, QCursor, QPainterPath
from PyQt5.QtCore import (
    Qt, QSize, QRect, QPoint, QPropertyAnimation, QTimer, QEvent
)

# ====================== 记忆化搜索 (Transposition Table) ======================

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# 全局置换表：Key -> (depth, value, flag)
TRANSPOSITION_TABLE = {}

def make_state_key(board, current_player, skipped_last, consecutive_moves, last_mover, switches_without_reduction, forbidden_move):
    """生成局面的哈希键（使用 Tuple 以利用 Python 字典的冲突处理）。"""
    # 将 board 转换为可哈希的 tuple 结构
    board_tuple = tuple(
        tuple(tuple(stack) for stack in row)
        for row in board
    )
    return (board_tuple, current_player, skipped_last, consecutive_moves, last_mover, switches_without_reduction, forbidden_move)


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
AI_SEARCH_DEPTH = 12    # 初始搜索深度（会自适应增加）
AI_MAX_SEARCH_TIME = 4.0  # 最大搜索时间（秒）
AI_MIN_DEPTH_TIME = 1.0   # 如果搜索时间小于这个值，继续加深


# ====================== 规则配置 ======================

class GameRules:
    def __init__(self):
        self.anti_backtracking = False
        self.n_move_rule = True
        self.n_move_limit = 6
        self.stalemate_rule = True
        self.stalemate_limit = 30

class SettingsDialog(QDialog):
    def __init__(self, rules: GameRules, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Special Rules Settings")
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

        # 1. Anti-Backtracking
        self.group_anti = QGroupBox("Anti-Backtracking")
        self.group_anti.setCheckable(True)
        self.group_anti.setChecked(self.rules.anti_backtracking)
        
        layout_anti = QVBoxLayout()
        lbl_anti = QLabel("Prevents moving a piece back to its previous position immediately.")
        lbl_anti.setStyleSheet("color: #666; font-size: 12px;")
        layout_anti.addWidget(lbl_anti)
        self.group_anti.setLayout(layout_anti)
        layout.addWidget(self.group_anti)

        # 2. N-Move Rule
        self.group_n = QGroupBox("N-Move Rule")
        self.group_n.setCheckable(True)
        self.group_n.setChecked(self.rules.n_move_rule)

        layout_n = QHBoxLayout()
        layout_n.addWidget(QLabel("Maximum consecutive moves allowed:"))
        self.sb_n_move = QSpinBox()
        self.sb_n_move.setRange(3, 20)
        self.sb_n_move.setValue(self.rules.n_move_limit)
        layout_n.addWidget(self.sb_n_move)
        layout_n.addStretch()
        self.group_n.setLayout(layout_n)
        layout.addWidget(self.group_n)

        # 3. Stalemate Draw
        self.group_s = QGroupBox("Stalemate Draw")
        self.group_s.setCheckable(True)
        self.group_s.setChecked(self.rules.stalemate_rule)

        layout_s = QHBoxLayout()
        layout_s.addWidget(QLabel("Maximum turns without stack reduction:"))
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

def check_unfavorable_3_stack_pattern(board: StackBoard, ai_player: Player) -> bool:
    """
    Check if the board has an unfavorable pattern for AI:
    - Exactly 3 stacks remain
    - One stack is in the "middle" position (adjacent to both other stacks)
    - The other two stacks are NOT adjacent to each other
    - The middle stack has AI's piece at the bottom
    - The two side stacks have opponent's pieces at the bottom
    
    This pattern is unfavorable because AI is trapped in the middle while
    the opponent controls both sides.
    """
    cells = get_nonempty_cells(board)
    if len(cells) != 3:
        return False
    
    # Get bottom pieces for each stack (only consider bottom, not top)
    stacks_info = []
    for r, c in cells:
        stack = board[r][c]
        if not stack:
            return False
        bottom = stack[0]  # Only check bottom piece
        stacks_info.append((r, c, bottom))
    
    opponent = 3 - ai_player
    
    # Find which stack is in the middle (adjacent to both other stacks)
    for i in range(3):
        middle_stack = stacks_info[i]
        other_stacks = [stacks_info[j] for j in range(3) if j != i]
        
        middle_pos = (middle_stack[0], middle_stack[1])
        other1_pos = (other_stacks[0][0], other_stacks[0][1])
        other2_pos = (other_stacks[1][0], other_stacks[1][1])
        
        # Check if middle stack is adjacent to both other stacks
        dist_to_other1 = abs(middle_pos[0] - other1_pos[0]) + abs(middle_pos[1] - other1_pos[1])
        dist_to_other2 = abs(middle_pos[0] - other2_pos[0]) + abs(middle_pos[1] - other2_pos[1])
        
        # Middle stack must be adjacent (distance 1) to both other stacks
        if dist_to_other1 != 1 or dist_to_other2 != 1:
            continue
        
        # Check if the other two stacks are NOT adjacent to each other
        dist_other1_other2 = abs(other1_pos[0] - other2_pos[0]) + abs(other1_pos[1] - other2_pos[1])
        if dist_other1_other2 == 1:
            # They are adjacent, so this is not the pattern we're looking for
            continue
        
        # Now check bottom pieces:
        # Middle stack should have AI's bottom piece
        # Both side stacks should have opponent's bottom pieces
        if (middle_stack[2] == ai_player and
            other_stacks[0][2] == opponent and
            other_stacks[1][2] == opponent):
            return True
    
    return False


def evaluate_board(board: StackBoard, ai_player: Player) -> float:
    """
    Simple heuristic:
    - Each stack: top is ai -> +height^2; top is opponent -> -height^2
    - Tallest stack: extra +/- 5 * height
    - Total piece difference: *(0.5) small correction
    - Penalty for unfavorable 3-stack pattern
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
    
    # Penalty for unfavorable 3-stack pattern
    if check_unfavorable_3_stack_pattern(board, ai_player):
        score -= 1000.0  # Significant penalty
    
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
    """带记忆化的 Alpha-Beta 搜索入口。"""
    # 1. 计算哈希键
    # 注意：需要根据 skipped_last 和 last_move_coords 预判 forbidden_move，以保持 Key 的唯一性
    f_move = None
    if rules.anti_backtracking and skipped_last and last_move_coords:
        src, dst = last_move_coords
        f_move = (dst, src)
        
    key = make_state_key(board, current_player, skipped_last, consecutive_moves, last_mover, switches_without_reduction, f_move)
    
    # 2. 查表
    if key in TRANSPOSITION_TABLE:
        entry_depth, entry_val, entry_flag = TRANSPOSITION_TABLE[key]
        if entry_depth >= depth:
            if entry_flag == TT_EXACT:
                return entry_val
            elif entry_flag == TT_LOWER:
                alpha = max(alpha, entry_val)
            elif entry_flag == TT_UPPER:
                beta = min(beta, entry_val)
            
            if alpha >= beta:
                return entry_val

    # 3. 计算
    alpha_orig = alpha
    val = _alpha_beta_compute(board, current_player, skipped_last, consecutive_moves, last_mover, last_move_coords, switches_without_reduction, rules, depth, alpha, beta, ai_player)
    
    # 4. 存表
    flag = TT_EXACT
    if val <= alpha_orig:
        flag = TT_UPPER
    elif val >= beta:
        flag = TT_LOWER
        
    TRANSPOSITION_TABLE[key] = (depth, val, flag)
    
    return val


def _alpha_beta_compute(board: StackBoard,
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
    """极大极小 + alpha-beta 剪枝（实际计算逻辑）。"""
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
    
    # 单一走法延伸策略（Single Extension）
    # 如果只有一个合法走法，这步是强制的，不应消耗搜索深度。
    # 这样可以让搜索在这些分支上看得更远。
    next_depth = depth - 1
    if len(legal_moves) == 1 and depth > 0:
        next_depth = depth

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
                                       next_depth,  # 使用计算好的 next_depth
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
                                       next_depth,  # 使用计算好的 next_depth
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

    def __init__(self, row: int, col: int, container):
        super().__init__(container)
        self.row = row
        self.col = col
        self.container = container
        self.setMinimumSize(QSize(100, 100))
        self.setMaximumSize(QSize(120, 120))
        self.setAttribute(Qt.WA_StyledBackground, True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if hasattr(self.container, 'cell_clicked'):
                self.container.cell_clicked(self.row, self.col)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        board = self.container.board
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

        selected = False
        if hasattr(self.container, 'selected_cell'):
            selected = (self.container.selected_cell == (self.row, self.col))

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
        self.setWindowTitle("Toppen")

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

        # 悔棋栈
        self.undo_stack = []

        self.init_ui()
        self.new_game()

    def init_ui(self):
        central = QWidget()
        central.setStyleSheet("background-color: white;")
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 顶部信息栏：按钮和 AI 评估状态
        top_layout = QHBoxLayout()
        
        button_style = """
            QPushButton {
                background-color: white; border: 1px solid #ccc; border-radius: 4px; 
                color: #333; font-size: 14px; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #f5f5f5; border-color: #999; }
            QPushButton:pressed { background-color: #e0e0e0; }
        """
        
        new_game_btn = QPushButton("New Game")
        new_game_btn.setStyleSheet(button_style)
        new_game_btn.clicked.connect(self.new_game)
        top_layout.addWidget(new_game_btn)
        
        rules_btn = QPushButton("Special Rules")
        rules_btn.setStyleSheet(button_style)
        rules_btn.clicked.connect(self.open_rules_dialog)
        top_layout.addWidget(rules_btn)
        
        custom_btn = QPushButton("Custom Game")
        custom_btn.setStyleSheet(button_style)
        custom_btn.clicked.connect(self.open_custom_game)
        top_layout.addWidget(custom_btn)
        
        load_btn = QPushButton("Load Game")
        load_btn.setStyleSheet(button_style)
        load_btn.clicked.connect(self.open_load_game)
        top_layout.addWidget(load_btn)
        
        top_layout.addStretch()
        
        self.indicator = QLabel()
        self.indicator.setFixedSize(20, 20)
        # 默认灰色，同时应用白底黑字 Tooltip 样式
        self.indicator.setStyleSheet("""
            QLabel {
                background-color: gray; 
                border-radius: 10px; 
                border: 1px solid #666;
            }
            QToolTip {
                background-color: white;
                color: black;
                border: 1px solid black;
            }
        """)
        self.indicator.setToolTip("AI Evaluation: Unknown")
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
            
        # 底部栏：悔棋按钮
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        self.undo_btn = QPushButton("↶")
        self.undo_btn.setFixedSize(20, 20)
        # 圆形按钮样式，与 indicator 保持一致的大小和圆角
        self.undo_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0; 
                border: 1px solid #999; 
                border-radius: 10px; 
                color: #333; 
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #666;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)
        self.undo_btn.setToolTip("Undo last move")
        self.undo_btn.clicked.connect(self.undo_last_human_move)
        bottom_layout.addWidget(self.undo_btn)
        
        main_layout.addLayout(bottom_layout)

        self.setCentralWidget(central)

        self.resize(550, 600)

    # ---------- 游戏控制 ----------

    def open_custom_game(self):
        dialog = BoardEditor(self)
        if dialog.exec_() == QDialog.Accepted:
            self.start_custom_game(dialog.board)
    
    def open_load_game(self):
        dialog = LoadGameDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.selected_board:
            self.start_custom_game(dialog.selected_board)

    def start_custom_game(self, custom_board: StackBoard):
        TRANSPOSITION_TABLE.clear()  # 清空记忆化缓存
        self.undo_stack.clear()      # 清空悔棋栈
        if self.animation is not None:
            self.animation.stop()
            self.animation = None
        if self.animation_widget is not None:
            self.animation_widget.hide()
            self.animation_widget.deleteLater()
            self.animation_widget = None

        # 深拷贝以防万一
        self.board = [[stack.copy() for stack in row] for row in custom_board]
        
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
        
        self.indicator.setStyleSheet("""
            QLabel {
                background-color: gray; 
                border-radius: 10px; 
                border: 1px solid #666;
            }
            QToolTip {
                background-color: white;
                color: black;
                border: 1px solid black;
            }
        """)
        self.indicator.setToolTip("AI Evaluation: Unknown")
        
        self.update_board_view()
        self.start_turn()

    def open_rules_dialog(self):
        dialog = SettingsDialog(self.rules, self)
        if dialog.exec_() == QDialog.Accepted:
            self.rules = dialog.get_rules()
            self.new_game()

    def new_game(self):
        TRANSPOSITION_TABLE.clear()  # 清空记忆化缓存
        self.undo_stack.clear()      # 清空悔棋栈
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
        self.indicator.setStyleSheet("""
            QLabel {
                background-color: gray; 
                border-radius: 10px; 
                border: 1px solid #666;
            }
            QToolTip {
                background-color: white;
                color: black;
                border: 1px solid black;
            }
        """)
        self.indicator.setToolTip("AI Evaluation: Unknown")
        
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
            name = "You" if winner == 1 else "Computer"
            QMessageBox.information(self, "Game Over",
                                    f"{name} wins!")
            # self.status_label.setText(f"游戏结束：玩家 {winner} 获胜。")
            self.new_game()
            return

        # 终局 1.5：底层控制获胜
        winner_bottom = check_bottom_control_winner(self.board)
        if winner_bottom is not None:
            self.game_over = True
            self.update_board_view()
            name = "You" if winner_bottom == 1 else "Computer"
            QMessageBox.information(self, "Game Over",
                                    f"{name} wins (controls all bottom pieces)!")
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
                QMessageBox.information(self, "Game Over",
                                        f"Draw: Neither player has legal moves.")
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

        # Adaptive depth search: start with initial depth, increase if search is fast
        start_time = time.time()
        best_move = None
        best_score = -inf
        current_depth = AI_SEARCH_DEPTH
        max_depth = 50  # Safety limit
        
        while current_depth <= max_depth:
            depth_start_time = time.time()
            
            move, score = choose_ai_move(self.board,
                                  self.current_player,
                                  self.skipped_last_turn,
                                  self.consecutive_moves,
                                  self.last_moving_player,
                                  forbidden_move,
                                  self.switches_without_reduction,
                                  self.rules,
                                  self.ai_player,
                                  current_depth)
            
            depth_time = time.time() - depth_start_time
            total_time = time.time() - start_time
            
            # Update best result (only if we found a valid move)
            if move is not None:
                best_move = move
                best_score = score
            elif best_move is None:
                # No valid move found at this depth, but we haven't found any move yet
                # Continue searching deeper in case a move appears
                if depth_time < AI_MIN_DEPTH_TIME and total_time < AI_MAX_SEARCH_TIME:
                    current_depth += 2
                    continue
                else:
                    # Time is running out or search is slow, stop
                    break
            
            # Check if we found a decisive result (win/loss)
            # If score indicates a clear win/loss, we can stop
            WIN_THRESHOLD = 5000.0
            if abs(best_score) > WIN_THRESHOLD:
                # Found a decisive result, use this depth
                break
            
            # Check if we've exceeded max time
            if total_time >= AI_MAX_SEARCH_TIME:
                # Time limit reached, use current best result
                break
            
            # If this depth search was very fast, try deeper
            if depth_time < AI_MIN_DEPTH_TIME:
                current_depth += 2  # Increase depth by 2 for efficiency
            else:
                # Search is taking longer, stop here
                break
        
        self.update_indicator(best_score)

        if best_move is None:
            self.skipped_last_turn = True
            self.current_player = 2 if self.current_player == 1 else 1
            self.start_turn()
            return

        src, dst = best_move
        self.animate_move(src, dst, next_player=2 if self.current_player == 1 else 1)

    def update_indicator(self, score: float):
        """根据 AI 评分更新指示器颜色。"""
        # 阈值设定：超过 5000 认为是必胜/必败
        WIN_THRESHOLD = 5000.0
        
        if score > WIN_THRESHOLD:
            # AI (Player 2) 必胜 -> 红色
            color = "red"
            tooltip = "Computer wins"
        elif score < -WIN_THRESHOLD:
            # Human (Player 1) 必胜 -> 绿色
            color = "#00cc00"
            tooltip = "Human wins"
        elif abs(score) < 0.1:
            # 平局 -> 蓝色
            color = "blue"
            tooltip = "Expected draw"
        else:
            # 局势不明 -> 灰色
            color = "gray"
            tooltip = "Uncertain"
            
        # 修改 tooltip 样式：移除自定义背景色，使用默认白底黑字（或系统默认）
        # 注意：QToolTip 的样式通常是全局设置，或者通过 stylesheet 设置 QToolTip
        # 这里我们只设置 QLabel 的背景色，tooltip 样式让其保持默认或通过全局样式控制
        # 如果需要强制白底黑字，可以在 app 级别设置，或者在这里给 label 设置 QToolTip 样式
        
        self.indicator.setStyleSheet(f"""
            QLabel {{
                background-color: {color}; 
                border-radius: 10px; 
                border: 1px solid #666;
            }}
            QToolTip {{
                background-color: white;
                color: black;
                border: 1px solid black;
            }}
        """)
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
        
        # 如果是人类玩家操作，记录快照以供悔棋
        # 注意：这里假设人类玩家是 1，或者非 AI 玩家
        if not (self.use_ai and self.current_player == self.ai_player):
            self.save_state_for_undo()
            
        self.animate_move((sr, sc), (r, c),
                          next_player=2 if self.current_player == 1 else 1)

    def save_state_for_undo(self):
        """保存当前状态快照到 undo_stack。"""
        snapshot = {
            'board': [[stack.copy() for stack in row] for row in self.board],
            'current_player': self.current_player,
            'skipped_last_turn': self.skipped_last_turn,
            'game_over': self.game_over,
            'consecutive_moves': self.consecutive_moves,
            'last_moving_player': self.last_moving_player,
            'last_move_record': self.last_move_record,
            'switches_without_reduction': self.switches_without_reduction,
            'forbidden_move': self.forbidden_move,
            # 保存指示器状态
            'indicator_style': self.indicator.styleSheet(),
            'indicator_tooltip': self.indicator.toolTip()
        }
        self.undo_stack.append(snapshot)

    def undo_last_human_move(self):
        """撤回人类的上一手操作（如果中间有AI操作也一并撤回）。"""
        if not self.undo_stack:
            return

        if self.animating:
            return
            
        snapshot = self.undo_stack.pop()
        
        self.board = snapshot['board']
        self.current_player = snapshot['current_player']
        self.skipped_last_turn = snapshot['skipped_last_turn']
        self.game_over = snapshot['game_over']
        self.consecutive_moves = snapshot['consecutive_moves']
        self.last_moving_player = snapshot['last_moving_player']
        self.last_move_record = snapshot['last_move_record']
        self.switches_without_reduction = snapshot['switches_without_reduction']
        self.forbidden_move = snapshot['forbidden_move']
        
        # 恢复后清理选中状态
        self.selected_cell = None
        self.animating = False
        self.pending_board_after_move = None
        self.anim_next_player = None
        
        # 停止可能的 AI 思考或动画
        if self.animation is not None:
            self.animation.stop()
            self.animation = None
        if self.animation_widget is not None:
            self.animation_widget.hide()
            self.animation_widget.deleteLater()
            self.animation_widget = None
            
        self.update_board_view()
        
        # 恢复 AI 指示器状态
        if 'indicator_style' in snapshot:
            self.indicator.setStyleSheet(snapshot['indicator_style'])
            self.indicator.setToolTip(snapshot['indicator_tooltip'])
        else:
            # 兼容旧快照（如果有）
            self.indicator.setStyleSheet("background-color: gray; border-radius: 10px; border: 1px solid #666;")
            self.indicator.setToolTip("AI Evaluation: Unknown")
        
        # 恢复后不需要 start_turn，因为肯定是回到了人类待机状态

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
            l_name = "You" if loser == 1 else "Computer"
            w_name = "You" if winner == 1 else "Computer"
            QMessageBox.information(self, "Game Over",
                                    f"{l_name} violated: Made {self.rules.n_move_limit} consecutive moves.\n"
                                    f"{w_name} wins!")
            self.new_game()
            return

        if self.rules.stalemate_rule and self.switches_without_reduction >= self.rules.stalemate_limit:
            self.game_over = True
            QMessageBox.information(self, "Game Over",
                                    f"Draw: {self.rules.stalemate_limit} consecutive turns without stack reduction.")
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


# ====================== 自定义局面编辑器 ======================

class BoardEditor(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Game")
        self.resize(600, 650)
        
        # 初始为空棋盘
        self.board = [[[] for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        
        # 拖拽状态
        self.dragging_piece = None  # 1 或 2
        self.drag_source = None     # 'supply' 或 (r, c)
        self.drag_label = None
        
        self.init_ui()
        
    def init_ui(self):
        self.setStyleSheet("font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; font-size: 14px;")
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # 1. Instructions
        instr_group = QGroupBox("")
        instr_layout = QVBoxLayout()
        lbl_instr = QLabel("Drag to place pieces.")
        lbl_instr.setStyleSheet("color: #555; line-height: 1.5;")
        instr_layout.addWidget(lbl_instr)
        instr_group.setLayout(instr_layout)
        layout.addWidget(instr_group)
        
        # 2. 棋盘区域
        grid_frame = QFrame()
        grid_frame.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 8px;")
        grid_layout = QGridLayout(grid_frame)
        grid_layout.setSpacing(6)
        grid_layout.setContentsMargins(15, 15, 15, 15)
        
        self.cells = []
        for r in range(BOARD_ROWS):
            row_widgets = []
            for c in range(BOARD_COLS):
                cell = StackWidget(r, c, self)
                # 安装事件过滤器以处理拖拽
                cell.installEventFilter(self)
                grid_layout.addWidget(cell, r, c)
                row_widgets.append(cell)
            self.cells.append(row_widgets)
            
        layout.addWidget(grid_frame)
        
        # 3. 供应区 & 垃圾桶
        supply_group = QGroupBox("")
        supply_layout = QHBoxLayout()
        supply_layout.setSpacing(20)
        
        # 辅助函数：创建不带标签的图标容器
        def create_supply_item(label_widget):
            container = QWidget()
            v_layout = QVBoxLayout(container)
            v_layout.setContentsMargins(0, 0, 0, 0)
            v_layout.setSpacing(5)
            v_layout.addWidget(label_widget, 0, Qt.AlignCenter)
            return container

        # 玩家1
        self.supply_p1 = QLabel()
        self.supply_p1.setFixedSize(64, 64)
        self.supply_p1.setStyleSheet("background-color: #eef; border: 2px dashed #aac; border-radius: 8px;")
        self.supply_p1.setAlignment(Qt.AlignCenter)
        self.draw_supply_icon(self.supply_p1, 1)
        self.supply_p1.installEventFilter(self)
        
        # 玩家2
        self.supply_p2 = QLabel()
        self.supply_p2.setFixedSize(64, 64)
        self.supply_p2.setStyleSheet("background-color: #fee; border: 2px dashed #caa; border-radius: 8px;")
        self.supply_p2.setAlignment(Qt.AlignCenter)
        self.draw_supply_icon(self.supply_p2, 2)
        self.supply_p2.installEventFilter(self)
        
        # 垃圾桶
        self.trash_bin = QLabel()
        self.trash_bin.setFixedSize(64, 64)
        self.trash_bin.setStyleSheet("background-color: #fff0f0; border: 2px solid #e57373; border-radius: 8px;")
        self.trash_bin.setAlignment(Qt.AlignCenter)
        self.draw_trash_icon(self.trash_bin)
        self.trash_bin.installEventFilter(self)

        supply_layout.addStretch()
        supply_layout.addWidget(create_supply_item(self.supply_p1))
        supply_layout.addWidget(create_supply_item(self.supply_p2))
        supply_layout.addSpacing(20)
        supply_layout.addWidget(create_supply_item(self.trash_bin))
        supply_layout.addStretch()
        
        supply_group.setLayout(supply_layout)
        layout.addWidget(supply_group)
        
        # 4. 底部栏（按钮）
        bottom_layout = QHBoxLayout()
        
        btn_save = QPushButton("Save")
        btn_save.setFixedSize(100, 36)
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; border: none; border-radius: 4px; color: white; font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #1565C0; }
        """)
        btn_save.clicked.connect(self.save_game)
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedSize(100, 36)
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5; border: 1px solid #ccc; border-radius: 4px; color: #333;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        btn_cancel.clicked.connect(self.reject)
        
        self.btn_start = QPushButton("Start Game")
        self.btn_start.setFixedSize(120, 36)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; border: none; border-radius: 4px; color: white; font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        self.btn_start.clicked.connect(self.check_and_start)
        
        bottom_layout.addStretch()
        bottom_layout.addWidget(btn_save)
        bottom_layout.addWidget(btn_cancel)
        bottom_layout.addWidget(self.btn_start)
        
        layout.addLayout(bottom_layout)

    def draw_trash_icon(self, label):
        pixmap = QPixmap(60, 60)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制垃圾桶
        painter.setPen(QPen(QColor("#e57373"), 2))
        painter.setBrush(Qt.NoBrush)
        
        cx, cy = 30, 30
        
        # 桶身 (梯形)
        body_top_w = 24
        body_bottom_w = 20
        body_h = 28
        
        path = QPainterPath()
        path.moveTo(cx - body_top_w/2, cy - body_h/2 + 4)
        path.lineTo(cx + body_top_w/2, cy - body_h/2 + 4)
        path.lineTo(cx + body_bottom_w/2, cy + body_h/2 + 4)
        path.lineTo(cx - body_bottom_w/2, cy + body_h/2 + 4)
        path.closeSubpath()
        painter.drawPath(path)
        
        # 桶盖
        lid_w = 28
        lid_h = 4
        painter.drawRoundedRect(int(cx - lid_w/2), int(cy - body_h/2), int(lid_w), int(lid_h), 2, 2)
        
        # 提手
        handle_w = 10
        handle_h = 3
        painter.drawArc(int(cx - handle_w/2), int(cy - body_h/2 - handle_h), int(handle_w), int(handle_h * 2), 0, 180 * 16)
        
        # 竖线纹理
        painter.drawLine(int(cx - 4), int(cy - body_h/2 + 8), int(cx - 3), int(cy + body_h/2))
        painter.drawLine(int(cx), int(cy - body_h/2 + 8), int(cx), int(cy + body_h/2))
        painter.drawLine(int(cx + 4), int(cy - body_h/2 + 8), int(cx + 3), int(cy + body_h/2))
        
        painter.end()
        label.setPixmap(pixmap)

    def draw_supply_icon(self, label, player):
        pixmap = QPixmap(60, 60)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = QRect(5, 5, 50, 50)
        if player == 1:
            base_color = QColor(180, 210, 255)
            border_color = QColor(20, 40, 90)
        else:
            base_color = QColor(255, 200, 200)
            border_color = QColor(120, 30, 30)
            
        painter.setBrush(base_color)
        painter.setPen(border_color)
        painter.drawRoundedRect(rect, 6, 6)
        
        # 绘制图标 (与 StackWidget 保持一致)
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
            
        painter.end()
        label.setPixmap(pixmap)

    def eventFilter(self, source, event):
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                if source == self.supply_p1:
                    self.start_drag(1, 'supply', source)
                    return True
                elif source == self.supply_p2:
                    self.start_drag(2, 'supply', source)
                    return True
                elif isinstance(source, StackWidget):
                    r, c = source.row, source.col
                    if self.board[r][c]:
                        piece = self.board[r][c].pop()
                        self.start_drag(piece, (r, c), source)
                        source.update()
                        return True
                        
        elif event.type() == QEvent.MouseMove:
            if self.dragging_piece is not None and self.drag_label:
                # 更新拖拽图标位置
                pos = event.globalPos()
                local_pos = self.mapFromGlobal(pos)
                self.drag_label.move(local_pos - QPoint(30, 30))
                return True
                
        elif event.type() == QEvent.MouseButtonRelease:
            if self.dragging_piece is not None:
                # 确定释放位置
                pos = event.globalPos()
                widget = QApplication.widgetAt(pos)
                
                dropped = False
                if isinstance(widget, StackWidget) and widget.container == self:
                    # 放置在格子上
                    self.board[widget.row][widget.col].append(self.dragging_piece)
                    widget.update()
                    dropped = True
                elif widget == self.trash_bin:
                    # 扔进垃圾桶
                    dropped = True # 视为成功处理（删除）
                
                if not dropped:
                    # 还原
                    if self.drag_source != 'supply':
                        r, c = self.drag_source
                        self.board[r][c].append(self.dragging_piece)
                        self.cells[r][c].update()
                
                self.end_drag()
                return True
                
        return super().eventFilter(source, event)

    def start_drag(self, piece, source, source_widget):
        self.dragging_piece = piece
        self.drag_source = source
        
        # 创建拖拽图标
        self.drag_label = QLabel(self)
        self.drag_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.draw_supply_icon(self.drag_label, piece)
        self.drag_label.resize(60, 60)
        self.drag_label.show()
        
        # 初始位置
        cursor_pos = self.mapFromGlobal(QCursor.pos())
        self.drag_label.move(cursor_pos - QPoint(30, 30))

    def end_drag(self):
        if self.drag_label:
            self.drag_label.deleteLater()
            self.drag_label = None
        self.dragging_piece = None
        self.drag_source = None

    def save_game(self):
        # 检查连通性
        if not is_connected(self.board):
            QMessageBox.warning(self, "Invalid Layout", "All pieces must be connected (adjacent up, down, left, or right).")
            return
            
        # 检查是否有棋子
        cells = get_nonempty_cells(self.board)
        if not cells:
            QMessageBox.warning(self, "Invalid Layout", "Board cannot be empty.")
            return
        
        # Get game name
        name, ok = QInputDialog.getText(self, "Save Game", "Enter game name:", text="Custom Game")
        if not ok or not name.strip():
            return
        
        name = name.strip()
        
        # 加载现有游戏列表
        games_file = "custom_games.json"
        games = []
        if os.path.exists(games_file):
            try:
                with open(games_file, 'r', encoding='utf-8') as f:
                    games = json.load(f)
            except:
                games = []
        
        # 检查名称是否已存在
        for i, game in enumerate(games):
            if game.get('name') == name:
                reply = QMessageBox.question(self, "Name Exists", 
                                            f"Game name '{name}' already exists. Overwrite?",
                                            QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    games[i] = {'name': name, 'board': self.board}
                    break
                else:
                    return
        else:
            # 添加新游戏
            games.append({'name': name, 'board': self.board})
        
        # 保存到文件
        try:
            with open(games_file, 'w', encoding='utf-8') as f:
                json.dump(games, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Error saving game: {str(e)}")

    def check_and_start(self):
        # 检查连通性
        if not is_connected(self.board):
            QMessageBox.warning(self, "Invalid Layout", "All pieces must be connected (adjacent up, down, left, or right).")
            return
            
        # 检查是否有棋子
        cells = get_nonempty_cells(self.board)
        if not cells:
            QMessageBox.warning(self, "Invalid Layout", "Board cannot be empty.")
            return
            
        self.accept()


# ====================== 加载游戏对话框 ======================

class GameThumbnail(QWidget):
    """游戏缩略图控件"""
    def __init__(self, game_data, dialog, parent=None):
        super().__init__(parent)
        self.game_data = game_data
        self.board = game_data['board']
        self.name = game_data['name']
        self.dialog = dialog
        self.is_selected = False
        self.setFixedSize(120, 150)
        self.setStyleSheet("""
            QWidget {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: white;
            }
            QWidget:hover {
                border: 2px solid #2196F3;
                background-color: #f0f8ff;
            }
        """)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # 绘制标题
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(painter.font())
        text_rect = QRect(5, 5, w - 10, 20)
        painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, self.name)
        
        # 绘制棋盘缩略图（确保是正方形）
        board_size = min(w - 20, h - 30 - 20)
        board_rect = QRect((w - board_size) // 2, 30, board_size, board_size)
        cell_size = board_size // BOARD_COLS
        
        # 绘制背景网格
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        for r in range(BOARD_ROWS + 1):
            y = board_rect.top() + r * cell_size
            painter.drawLine(board_rect.left(), y, board_rect.right(), y)
        for c in range(BOARD_COLS + 1):
            x = board_rect.left() + c * cell_size
            painter.drawLine(x, board_rect.top(), x, board_rect.bottom())
        
        # 绘制棋子（用小方块表示，考虑堆叠）
        base_piece_size = int(cell_size * 0.5)  # 基础大小，为堆叠留出空间
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                stack = self.board[r][c]
                if stack:
                    # 计算堆叠的起始位置（居中）
                    cell_left = board_rect.left() + c * cell_size
                    cell_top = board_rect.top() + r * cell_size
                    center_x = cell_left + cell_size // 2
                    center_y = cell_top + cell_size // 2
                    
                    # 堆叠数量
                    stack_height = len(stack)
                    
                    # 堆叠偏移量（每层向右下偏移）
                    offset_step = 1.5
                    max_offset = (stack_height - 1) * offset_step
                    
                    # 确保堆叠不会超出格子
                    if max_offset + base_piece_size > cell_size * 0.8:
                        offset_step = (cell_size * 0.8 - base_piece_size) / max(1, stack_height - 1)
                    
                    # 从底层到顶层绘制
                    for idx, player in enumerate(stack):
                        # 计算偏移（底层在左上，顶层在右下）
                        offset_x = idx * offset_step
                        offset_y = idx * offset_step
                        
                        # 计算当前层的位置
                        piece_x = center_x - base_piece_size // 2 + offset_x
                        piece_y = center_y - base_piece_size // 2 + offset_y
                        
                        rect = QRect(
                            int(piece_x),
                            int(piece_y),
                            base_piece_size,
                            base_piece_size
                        )
                        
                        if player == 1:
                            # 蓝色（玩家1）
                            base_color = QColor(180, 210, 255)
                            border_color = QColor(20, 40, 90)
                        else:
                            # 红色（玩家2）
                            base_color = QColor(255, 200, 200)
                            border_color = QColor(120, 30, 30)
                        
                        # 底层稍微暗一些，顶层亮一些
                        depth = stack_height - 1 - idx
                        factor = 1.0 - 0.1 * depth
                        factor = max(0.7, factor)
                        color = QColor(
                            int(base_color.red() * factor),
                            int(base_color.green() * factor),
                            int(base_color.blue() * factor)
                        )
                        
                        painter.setBrush(color)
                        painter.setPen(border_color)
                        painter.drawRoundedRect(rect, 2, 2)
        
        # 如果被选中，绘制蒙版和选中图标
        if self.is_selected:
            # 绘制半透明蓝色蒙版
            overlay_color = QColor(33, 150, 243, 120)  # 半透明蓝色
            painter.setBrush(overlay_color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(self.rect(), 8, 8)
            
            # 绘制选中图标（右上角的勾选标记）
            check_size = 24
            check_x = w - check_size - 5
            check_y = 5
            
            # 绘制圆形背景
            check_bg_rect = QRect(check_x, check_y, check_size, check_size)
            painter.setBrush(QColor(33, 150, 243))  # 蓝色背景
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(check_bg_rect)
            
            # 绘制白色勾选标记
            pen = QPen(QColor(255, 255, 255), 3)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            # 勾选标记的路径（更平滑的勾）
            check_path = QPainterPath()
            center_x = check_x + check_size // 2
            center_y = check_y + check_size // 2
            check_path.moveTo(center_x - 4, center_y)
            check_path.lineTo(center_x - 1, center_y + 3)
            check_path.lineTo(center_x + 4, center_y - 2)
            painter.drawPath(check_path)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dialog.select_game(self.game_data)


class LoadGameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Game")
        self.resize(600, 500)
        self.selected_board = None
        
        self.init_ui()
        self.load_games()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Select a game to load")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        layout.addWidget(title)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #ddd; border-radius: 8px; }")
        
        # 内容区域（网格布局）
        content_widget = QWidget()
        self.grid_layout = QGridLayout(content_widget)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setContentsMargins(15, 15, 15, 15)
        # 确保靠左对齐
        self.grid_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        scroll.setWidget(content_widget)
        layout.addWidget(scroll)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedSize(100, 36)
        btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #f5f5f5; border: 1px solid #ccc; border-radius: 4px; color: #333;
            }
            QPushButton:hover { background-color: #e0e0e0; }
        """)
        btn_cancel.clicked.connect(self.reject)
        
        self.btn_load = QPushButton("Load")
        self.btn_load.setFixedSize(100, 36)
        self.btn_load.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; border: none; border-radius: 4px; color: white; font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:pressed { background-color: #3d8b40; }
        """)
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self.accept)
        
        button_layout.addWidget(btn_cancel)
        button_layout.addWidget(self.btn_load)
        layout.addLayout(button_layout)
        
        self.content_widget = content_widget
        self.selected_game = None
    
    def load_games(self):
        games_file = "custom_games.json"
        games = []
        
        if os.path.exists(games_file):
            try:
                with open(games_file, 'r', encoding='utf-8') as f:
                    games = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "Load Failed", f"Error reading game list: {str(e)}")
                return
        
        if not games:
            no_games_label = QLabel("No saved games")
            no_games_label.setAlignment(Qt.AlignCenter)
            no_games_label.setStyleSheet("color: #999; font-size: 14px; padding: 40px;")
            self.grid_layout.addWidget(no_games_label, 0, 0, 1, 4)
            return
        
        # 每行显示4个，从上到下、从左到右排列
        cols_per_row = 4
        for i, game_data in enumerate(games):
            row = i // cols_per_row
            col = i % cols_per_row
            thumbnail = GameThumbnail(game_data, self, self.content_widget)
            # 确保每个widget靠左对齐
            self.grid_layout.addWidget(thumbnail, row, col, Qt.AlignLeft | Qt.AlignTop)
    
    def select_game(self, game_data):
        self.selected_game = game_data
        self.selected_board = game_data['board']
        self.btn_load.setEnabled(True)
        
        # 更新所有缩略图的选中状态
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if item:
                widget = item.widget()
                if isinstance(widget, GameThumbnail):
                    if widget.game_data == game_data:
                        widget.is_selected = True
                        widget.setStyleSheet("""
                            QWidget {
                                border: 3px solid #2196F3;
                                border-radius: 8px;
                                background-color: white;
                            }
                        """)
                    else:
                        widget.is_selected = False
                        widget.setStyleSheet("""
                            QWidget {
                                border: 2px solid #ddd;
                                border-radius: 8px;
                                background-color: white;
                            }
                            QWidget:hover {
                                border: 2px solid #2196F3;
                                background-color: #f0f8ff;
                            }
                        """)
                    widget.update()


def main():
    app = QApplication(sys.argv)
    window = ToppenWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()