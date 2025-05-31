# Goal
めぼしいユーザーのシャープレシオを求めてソートすることを目指す

## leaderboard.json
get_leaderboard_json.shを使う

```
(base) shindo@shindo-tp:~/hyperliquid-bot/user_sharpe$ cat leaderboard.json  | grep ethAddress | wc -l
17155
```

17155アドレスある
このアドレスから、
- 口座履歴
- 取引履歴
を求める


## 口座履歴
