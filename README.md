# Toppen Game Rules

Toppen is an abstract strategy game based on a stacking mechanism. The goal is to move and stack pieces to eventually occupy the highest point.

## Basic Rules

1.  **Board and Initial Setup**
    *   The game is played on a 4x4 grid.
    *   There are 10 pieces total (5 for each player).
    *   At the start, pieces are randomly stacked to form a connected initial layout.

2.  **Movement Mechanism**
    *   Players take turns.
    *   Players can only move **their own colored pieces that are on top of a stack**.
    *   Pieces can only move to **adjacent** (up, down, left, right) cells that **already contain pieces** (cannot move to empty cells).
    *   **Connectivity Principle**: After a move, all remaining piece stacks on the board must remain **connected**. If a move would split the board into disconnected parts, that move is illegal.

3.  **Passing a Turn**
    *   If a player has no legal moves when it's their turn (e.g., all their pieces are buried, or any move would break connectivity), they must **pass** their turn.

4.  **Victory Conditions**
    *   **Summit Victory (Primary Goal)**: When all pieces on the board merge into a **single stack**, the game ends immediately. The player whose piece is on **top** of that stack wins.
    *   **Bottom Control Victory (Domination)**: If the **bottom** piece of **all** non-empty cells belongs to the same player, that player wins immediately.
    *   **Tower Victory (Stalemate Resolution)**: If both players pass consecutively (i.e., neither has legal moves), the game ends. The player whose piece is on top of the **tallest** stack wins.

---

## Special Rules

To increase strategic depth and prevent infinite loops, the game includes the following special rules. **These rules can be enabled/disabled or adjusted in the "Special Rules" menu in the game interface.**

1.  **Anti-Backtracking**
    *   **Trigger**: When the opponent is forced to pass.
    *   **Rule**: You cannot immediately move a piece back to its previous position.
    *   **Purpose**: Prevents infinite loops when one player has no moves and the other moves the same piece back and forth.
    *   *Disabled by default.*

2.  **N-Move Rule**
    *   **Rule**: If a player makes **N consecutive moves** (default N=5, meaning the opponent was forced to pass N-1 times), that player **loses**.
    *   **Purpose**: Prevents a player from exploiting an opponent's lack of moves to stall the game, forcing players to end the game quickly.
    *   *Enabled by default, default limit is 6 moves.*

3.  **Stalemate Draw**
    *   **Rule**: If after **N consecutive turns** (default N=20, alternating between players, consecutive moves by one player don't count), the **number of stacks on the board has not decreased**, the game is declared a **draw**.
    *   **Purpose**: Prevents the game from entering a deadlock where neither player can win.
    *   *Enabled by default, default limit is 30 turns.*

## Interface Guide
*   **Top-right indicator**: AI position evaluation indicator.
    *   🟢 **Green**: Human player has a winning strategy.
    *   🔴 **Red**: Computer player has a winning strategy.
    *   🔵 **Blue**: AI predicts a draw.
    *   ⚪ **Gray**: Current position is uncertain.
