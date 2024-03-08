线性筛选

```c++
#include<cstring>
#define MAXN 100005
#define MAXL 1299710
int prime[MAXN];
int check[MAXL];
int main() {
    int tot = 0;
    memset(check, 0, sizeof(check));
    for (int i = 2; i < MAXL; ++i) {
        if (!check[i]) {
            prime[tot++] = i;
        }
        for (int j = 0; j < tot; ++j) {
            if (i * prime[j] > MAXL) {
                break;
            }
            check[i * prime[j]] = 1;
            if (i % prime[j] == 0) {
                break;
            }
        }
    }
}
```

递归 to 栈

```c++
#include <bits/stdc++.h>

using namespace std;

struct sta {
    char fr, to, mid;
    int n, line;

    inline sta(char fr_ = 0, char to_ = 0, char mid_ = 0, int n_ = 0, int line_ = 0) :
            fr(fr_), to(to_), mid(mid_), n(n_), line(line_) {
    }
};

void move(char fr, char to, char mid, int n) {
    if (n == 0) return;//1
    move(fr, mid, to, n - 1);//2
    cout << fr << " -> " << to << "\n";//3
    move(mid, to, fr, n - 1);//4
    //5
}

int n, curstk;


sta *stk;

inline void work() {
    stk[++curstk] = sta('a', 'c', 'b', n, 1);
    while (curstk) {
        sta now = stk[curstk--];
        if (now.n == 0) continue;
        if (now.line == 1) ++now.line;
        if (now.line == 2) {
            ++now.line;
            stk[++curstk] = now;
            stk[++curstk] = sta(now.fr, now.mid, now.to, now.n - 1, 1);
            continue;
        }
        if (now.line == 3) {
            cout << now.fr << " -> " << now.to << "\n";
            ++now.line;
        }
        if (now.line == 4) {
            ++now.line;
            stk[++curstk] = now;
            stk[++curstk] = sta(now.mid, now.to, now.fr, now.n - 1, 1);
            continue;
        }
        if (now.line == 5)
            continue;
    }
}

int main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    stk = new sta[n + 10];
    move('a', 'c', 'b', n);
    cout.flush();
    return 0;
}
```

