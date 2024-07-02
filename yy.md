[TOC]



## 进制问题

### 十进制转x进制

```c++
signed main() {
    int n, x, cnt = 0;
    char s[105];
    cin >> n >> x;
    while (n > 0) {
        int w = n % x;
        if (w < 10) {
            s[cnt++] = w + '0';
        } else {
            s[cnt++] = w - 10 + 'A';
        }
        n /= x;
    }
    for (int i = cnt - 1; i >= 0; i--) {
        cout << s[i];
    }
}
```

### x进制转十进制

```c++
signed main() {
    int n, x, ans = 0;
    char s[105];
    cin >> s >> x;
    n = strlen(s);
    for (int i = 0; i < n; i++) {
        ans = ans * x;
        if (s[i] >= '0' && s[i] <= '9') {
            ans += s[i] - '0';
        } else {
            ans += s[i] - 'A' + 10;
        }
    }
    cout << ans << endl;
}
```

### x进制转十进制

```c++
signed main() {
    int n, x, y, ans = 0;
    char s[105];
    cin >> s >> x >> y;
    n = strlen(s);
    // 转换为十进制
    for (int i = 0; i < n; i++) {
        ans = ans * x;
        if (s[i] >= '0' && s[i] <= '9') {
            ans += s[i] - '0';
        } else {
            ans += s[i] - 'A' + 10;
        }
    }
    // 转换为y进制
    char res[105];
    int cnt = 0;
    while (ans) {
        int t = ans % y;
        if (t >= 0 && t <= 9) {
            res[cnt++] = t + '0';
        } else {
            res[cnt++] = t - 10 + 'A';
        }
        ans /= y;
    }
    for (int i = cnt - 1; i >= 0; i--) {
        cout << res[i];
    }
    cout << endl;
}
```

## 排版问题

## 日期问题

