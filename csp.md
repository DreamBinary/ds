```c++
#include "bits/stdc++.h"
using namespace std;
#define int long long
#define pii pair<int, int>
#define IOS ios::sync_with_stdio(0); cin.tie(0); cout.tie(0);
#define endl '\n'
#define vi vector<int>
#define debug(x) cout << #x << " = " << x << endl;
#define fori(i, n) for(int i = 0; i < n; i++)
const int N = 1e9 + 10;
signed main() {
    IOS
    int n[3] = {0, 0, 0};
    fori(i, 3) { cin >> n[i]; }
    return 0;
}
```

[TOC]

<div STYLE="page-break-after: always;"></div>



### STL

```c++
vector
    vector<int> v;
    vector<int> vv = vector(v.begin(), v.end());
    v.push_back(1); v.emplace_back(1); // add element
    v.pop_back(); // remove last element
    v.insert(v.begin(), 1); // insert element
    v.erase(v.begin()); // erase element
    bool empty = v.empty(); // check if empty
    find(v.begin(), v.end(), 1) != v.end(); // find element

string
    s.begin(); s.end(); // 迭代器
    s.rbegin(); s.rend(); // 反向迭代器
    s.size();  // 长度
    auto i = s.find('a'); // 查找字符
    if (i != string::npos) {} // npos是string的静态成员，表示查找失败
    string ss = s.substr(1, 2); // 截取子串 param1: 起始位置 param2: 截取长度
    string sss = string(s, 1, 2)
    x = stoi(s.substr(0, d));
    y = stoi(s.substr(d + 1));


queue
    queue<int> q;
    q.push(1);
    q.pop();
    q.size();
    q.front();
    q.back();
    q.empty();
    
stack
    stack<int> s;
    s.push(1);
    s.pop();
    s.size();
    s.top();
    s.empty();
    
priority_queue
    priority_queue<int> pq;
    pq.push(1);
    pq.pop();
    pq.size();
    pq.top();
    pq.empty();

deque
    deque<int> dq;
    dq.push_back(1);
    dq.push_front(2);
    dq.pop_back();
    dq.pop_front();
    dq.front();
    dq.back();
    dq.size();
    dq.empty();
    dq.clear();

set
    set<int> s;
    unordered_set<int> us;
    s.insert(1);
    s.erase(1);
    auto addr = s.find(1);  // return iterator
    size_t cnt = s.count(1);  // return 0 or 1

map
    map<int, int> m;
    unordered_map<int, int> um;
    m[1] = 2;
    m.erase(1);
    auto addr = m.find(1);  // return iterator

vector<int> v1, v2, v;
set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), inserter(v, v.begin()));
set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), inserter(v, v.begin()));
set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), inserter(v, v.begin()));
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });

bitset, 圧位
    bitset<10000> s;
    ~, &, |, ^
    >>, <<
    ==, !=
    []

    count()  返回有多少个1

    any()  判断是否至少有一个1
    none()  判断是否全为0

    set()  把所有位置成1
    set(k, v)  将第k位变成v
    reset()  把所有位变成0
    flip()  等价于~
    flip(k) 把第k位取反
// __builtin_popcount(x) 返回x的二进制表示中1的个数
// __builtin_ctz(x) 返回x的二进制表示中从右往左第一个1的位置
// __builtin_clz(x) 返回x的二进制表示中从左往右第一个1的位置
```

```c++
素数
int p[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 93, 97, 101,103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211};

// int -->> -2147483648 to 2147483647 (1e9)
// unsigned int -->> 0 to 4294967295 (1e10)
// ll  -->> -9223372036854775808 to 9223372036854775807 (1e18)
// unsigned ll -->> 0 to 18446744073709551615 (1e20)
```

### 离散化

```c++
vector<int> all;
int get(int x) {
    return lower_bound(all.begin(), all.end(), x) - all.begin();
}

signed main() {
    vector<int> a = {100, 200, 300, 400, 400, 500};
    for(auto x : a) {
        all.push_back(x);
    }
    sort(all.begin(), all.end());
    all.erase(unique(all.begin(), all.end()), all.end());
    // unique()函数将重复的元素移到后面，返回指向第一个重复元素的迭代器，然后erase()函数将重复元素删除
    int n = a.size();
    for(int i = 0; i < n; i++) {
        cout << get(a[i]) << " ";
    } // 0 1 2 3 3 4
}

// 线段树
// 注意离散化要加入前后的点
// coordinates.push_back(l);
//  coordinates.push_back(r);
// if (l != 1) coordinates.push_back(l-1);    
// if (r != m) coordinates.push_back(r+1);
```

### 背包问题

```c++
// n -->> 物品数量
// c -->> 背包容量
// w -->> 物品重量
// v -->> 物品价值
// 0-1背包问题 -->> 每个物品只能取一次
void knapsack(int n, int c, int w[], int v[]) {
    int dp[n + 1][c + 1];
    memset(dp, 0, sizeof(dp));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= c; j++) {
            if (j < w[i]) {
                dp[i][j] = dp[i - 1][j];
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]);
            }
        }
    }
    cout << dp[n][c] << endl;
}

// 完全背包问题 -->> 每个物品可以取无限次
void completeKnapsack(int n, int c, int w[], int v[]) {
    int dp[n + 1][c + 1];
    memset(dp, 0, sizeof(dp));
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= c; j++) {
            if (j < w[i]) {
                dp[i][j] = dp[i - 1][j];
            } else {
                for (int k = 0; k * w[i] <= j; k++) {
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - k * w[i]] + k * v[i]);
                }
            }
        }
    }
    cout << dp[n][c] << endl;
}
```

### 单调栈 找出每个数左边离它最近的比它大/小的数

```c++
int stk[100010], top = 0;

int main() {
    int n;
    cin >> n;
    for (int i = 1; i <= n; i++) {
        int x;
        cin >> x;
        while (top && stk[top] >= x) top--;
        if (top) cout << stk[top] << ' ';
        else cout << -1 << ' ';
        stk[++top] = x;
    }
    return 0;
}
// example input
// 5
// 3 4 1 5 6
// example output
// -1 3 -1 1 5
```

### 单调队列 找出滑动窗口中的最大值/最小值

```c++
int q[1000005], a[1000005];
int hh = 0, tt = -1;

int main() {
    int n, k;
    cin >> n >> k;
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }
    // 找出滑动窗口中的最大值
    for (int i = 0; i < n; i++) {
        if (hh <= tt && i - k + 1 > q[hh]) {
            hh++;
        }
        while (hh <= tt && a[q[tt]] >= a[i]) {
            tt--;
        }
        q[++tt] = i;
        if (i >= k - 1) {
            cout << a[q[hh]] << ' ';
        }
    }
    cout << endl;
    // 找出滑动窗口中的最小值
    hh = 0, tt = -1;
    for (int i = 0; i < n; i++) {
        if (hh <= tt && i - k + 1 > q[hh]) {
            hh++;
        }
        while (hh <= tt && a[q[tt]] <= a[i]) {
            tt--;
        }
        q[++tt] = i;
        if (i >= k - 1) {
            cout << a[q[hh]] << ' ';
        }
    }
    cout << endl;
    return 0;
}
// example input
// 8 3
// 1 3 -1 -3 5 3 6 7
// example output
// 3 3 5 5 6 7 7
```

### 差分

```c++
前缀和数组 a[]
差分数组 b[]
    
对a数组区间[l,r]同时加上c的操作可转化为：
void insert(int l, int r, int c) {
    b[l] += c;
    b[r+1] -= c;
}
while(m--) {
    int l,r,c;
    scanf("%d%d%d",&l,&r,&c);
    insert(l,r,c);
}

对b数组求前缀和即可得到原数组a：
for(int i = 1; i <= n; i++) {
    b[i] += b[i-1];
    printf("%d ",b[i]);
}

S[x2, y2] = S[x1 - 1, y2] - S[x2, y1 - 1] + S[x1 - 1, y1 - 1]
S[x1, y1] += c, S[x2 + 1, y1] -= c, S[x1, y2 + 1] -= c, S[x2 + 1, y2 + 1] += c
```

### 线性筛素数

```c++
int primes[N], cnt;
bool st[N];

// primes -->> store the prime numbers
// cnt -->> the number of prime numbers
void get_primes(int n) {
    for (int i = 2; i <= n; i++) {
        if (!st[i]) primes[cnt++] = i;
        for (int j = 0; primes[j] <= n / i; j++) {
            st[primes[j] * i] = true;
            if (i % primes[j] == 0) break;
        }
    }
}
```

### 并查集

```c++
const int N = 1e5 + 10;
int fa[N];

inline void init(int n) {
    for (int i = 1; i <= n; i++) {
        fa[i] = i;
    }
}

int find(int x) {
    if (fa[x] == x) {
        return x;
    }
    return fa[x] = find(fa[x]);
}

void merge(int x, int y) {
    int fx = find(x);
    int fy = find(y);
    if (fx != fy) {
        fa[fx] = fy;
    }
}

void clear(int n) {
    for (int i = 1; i <= n; i++) {
        fa[i] = find(i);
        cout << fa[i] << " ";
    }
    cout << endl;

}
```

### 快速幂

```c++
// a^b % p  a -->> base, b -->> exponent, p -->> mod
int ksm(int a, int b, int p) {
    int ans = 1;
    while (b) {
        if (b & 1) ans = ans * a % p;
        a = a * a % p;
        b >>= 1;
    }
    return ans;
}
```

### 矩阵计算

```c++
struct Matrix {
    int r, c;
    vector<vector<int>> v;

    Matrix(int r, int c) : r(r), c(c) {
        v.resize(r, vector<int>(c));
    }
    void input() {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                cin >> v[i][j];
            }
        }
    }

    void output() {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                cout << v[i][j] << " ";
            }
            cout << endl;
        }
    }
    
    Matrix operator+(const Matrix &other) {
        Matrix sum(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                sum.v[i][j] = v[i][j] + other.v[i][j];
            }
        }
        return sum;
    }
    
    Matrix operator-(const Matrix &other) {
        Matrix diff(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                diff.v[i][j] = v[i][j] - other.v[i][j];
            }
        }
        return diff;
    }
    
    Matrix operator*(int scalar) {
        Matrix product(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                product.v[i][j] = v[i][j] * scalar;
            }
        }
        return product;
    }
    
    Matrix operator*(const Matrix &other) {
        Matrix product(r, other.c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < other.c; j++) {
                for (int k = 0; k < c; k++) {
                    product.v[i][j] += v[i][k] * other.v[k][j];
                }
            }
        }
        return product;
    }

    
    Matrix operator/(int scalar) {
        Matrix quotient(r, c);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                quotient.v[i][j] = v[i][j] / scalar;
            }
        }
        return quotient;
    }
};
```

### 字符串Hash

```c++
ll h[N], p[N];
ll mod = 1e9 + 7;

// l, r 为区间 return [l, r]的hash值
ll get(int l, int r) {
    return (h[r] - h[l - 1] * p[r - l + 1] % mod + mod) % mod;
}

// 初始化
void hash(string s) {
    int n = s.size();
    p[0] = 1;
    for (int i = 1; i <= n; i++) {
        h[i] = (h[i - 1] * 131 + s[i - 1]) % mod;
        p[i] = p[i - 1] * 131 % mod;
    }
}
```

### 分解质因数

```c++
void divide(int x) {
    for (int i = 2; i <= x / i; i++)
        if (x % i == 0) {
            int s = 0;
            while (x % i == 0) x /= i, s++;
            cout << i << ' ' << s << endl;
        }
    if (x > 1) cout << x << ' ' << 1 << endl;
    cout << endl;
}
```

### 递推求组合数

```c++
const int N = 10;
const int mod = 1e9 + 7;
int c[N][N];
int main() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++)
            if (!j) c[i][j] = 1;
            else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
    cout << c[5][2] << " " << c[3][2] << endl;
}
```

### 乘法逆元

```c++
// 5×3≡1(mod14)，我们称此时的3为5关于1模14的乘法逆元。 a = 5, p = 14
ll exgcd(ll a, ll b, ll &x, ll &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    ll d = exgcd(b, a % b, y, x);
    y = y - a / b * x;
    return d;
}
ll inv(ll a, ll mod) {
    ll x, y;
    exgcd(a, mod, x, y);
    return (x % mod + mod) % mod;
}
```

### 卡特兰数

```c++
给定n个0和n个1，它们按照某种顺序排成长度为2n的序列，满足任意前缀中0的个数都不少于1的个数的序列的数量为： Cat(n) = C(2n, n) / (n + 1) = C(2n, n) - C(2n, n + 1)
```

### ST表 查询区间最值

```c++
const int N = 1e5 + 10;
int pre[N][30], a[N];
void init(int n) {
    for (int i = 1; i <= n; i++) pre[i][0] = a[i];
    for (int j = 1; j <= 20; j++) {
        for (int i = 1; i + (1 << j) - 1 <= n; i++) {
            pre[i][j] = max(pre[i][j - 1], pre[i + (1 << (j - 1))][j - 1]);
        }
    }
}
int query(int l, int r) {
    int k = 31 - __builtin_clz(r - l + 1);
    return max(pre[l][k], pre[r - (1 << k) + 1][k]);
    return min(pre[l][k], pre[r - (1 << k) + 1][k]);
}
int main() {
    int n = 10;
    for (int i = 1; i <= n; i++) a[i] = rand() % 100;
    for (int i = 1; i <= n; i++) cout << a[i] << " ";
    cout << endl;
    init(n);
    cout << query(1, 5) << endl;
}
```

### Trie树 字符串的快速查询

```c++

int son[100010][26], val[100010], idx;
// son[i][j]表示第i个节点的第j个儿子是谁
// val[i]表示第i个节点的值是多少
void insert(const string &s, int x) {
    int p = 0;
    for (char i: s) {
        int u = i - 'a';
        if (!son[p][u]) son[p][u] = ++idx;
        p = son[p][u];
    }
    val[p] = x;
}
int query(const string &s) {
    int p = 0;
    for (char i: s) {
        int u = i - 'a';
        if (!son[p][u]) return -1;
        p = son[p][u];
    }
    return val[p];
}
```

### KMP

```c++
int nxt[N];
void getnxt(string s) {
    int n = s.size();
    nxt[0] = -1;
    for (int i = 0, j = -1; i < n; i++) {
        while (j != -1 && s[i] != s[j]) j = nxt[j];
        nxt[i + 1] = ++j;
    }
}

vector<int> kmp(string str, string substr) {
    vector<int> res;
    getnxt(substr);
    int n = str.size(), m = substr.size();
    for (int i = 0, j = 0; i < n; i++) {
        while (j != -1 && str[i] != substr[j]) j = nxt[j];
        if (++j == m) {
            res.push_back(i - m + 1);
            j = nxt[j];
        }
    }
    return res;
}
```

### 树状数组 单点修改，区间查询

```c++
int t[N];
void add_max(int x, int v) {
    for (; x < N; x += x & -x) t[x] = max(t[x], v);
}

int find_max(int x) {
    int res = 0;
    for (; x; x -= x & -x) res = max(res, t[x]);
    return res;
}

void add_sum(int x, int v) {
    for (; x < N; x += x & -x) t[x] += v;
}

int find_sum(int x) {
    int res = 0;
    for (; x; x -= x & -x) res += t[x];
    return res;
}

int find_sum(int l, int r) {
    return find_sum(r) - find_sum(l - 1);
}
```

### 树状数组 区间修改，区间求和

```c++
ll c[2][N], sum[N];
// c[0]存储的是原数组，c[1]存储的是前缀和
int n;
void add_inter(int k, int x, int d) {
    for (; x <= n; x += x & -x) c[k][x] += d;
}
void add(int l, int r, int d) {
    add_inter(0, l, d);
    add_inter(0, r + 1, -d);
    add_inter(1, l, d * (l - 1));
    add_inter(1, r + 1, -d * r);
}
ll ask(int k, int x) {
    ll res = 0;
    for (; x; x -= x & -x) res += c[k][x];
    return res;
}
ll query(int l, int r) {
    return ask(0, r) * r - ask(1, r) - ask(0, l - 1) * (l - 1) + ask(1, l - 1);
}
int main() {
    n = 5;
    for (int i = 1; i <= n; i++) {
        add(i, i, i);
    }
    for (int i = 1; i <= n; i++) {
        cout << ask(0, i) << " " << ask(1, i) << endl;
//        1 0
//        2 1
//        3 3
//        4 6
//        5 10
    }
    for (int i = 1; i <= n; i++) {
        cout << query(i, i) << " ";
//        1 2 3 4 5
    }
    cout << endl << query(1, n) << endl;
//    15
}
```

### 线段树 单点修改，区间查询

```c++
int s[N]; // the original array
struct Node {
    int index, val;
    Node operator+(const Node &a) const {
//        return val <= a.val ? *this : a; // minimum
//        return val >= a.val ? *this : a; // maximum
        return {index, val + a.val}; // sum
//        return {index, val * a.val}; // product
    }
} t[N];
void build(int v, int l, int r) {
    if (l == r) {
        t[v] = {l, s[l]};
        return;
    }
    int m = (l + r) / 2;
    build(v << 1, l, m);
    build(v << 1 | 1, m + 1, r);
    t[v] = t[v << 1] + t[v << 1 | 1];
}
void update(int v, int l, int r, int pos, int val) {
    if (l == r) {
        t[v] = {l, val};
        return;
    }
    int m = (l + r) / 2;
    if (pos <= m) update(v << 1, l, m, pos, val);
    else update(v << 1 | 1, m + 1, r, pos, val);
    t[v] = t[v << 1] + t[v << 1 | 1];
}
// return the xxx value in the range [ql, qr]
// v is the index of the current node
// l and r are the left and right boundaries of the current node
// ql and qr are the left and right boundaries of the query range
Node query(int v, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return t[v];
    int m = (l + r) / 2;
    if (qr <= m) return query(v << 1, l, m, ql, qr);
    if (ql > m) return query(v << 1 | 1, m + 1, r, ql, qr);
    return query(v << 1, l, m, ql, qr) + query(v << 1 | 1, m + 1, r, ql, qr);
}
signed main() {
    int n = 10;
    for (int i = 1; i <= n; i++) s[i] = i;
    int v = 1, l = 1, r = n;
    build(v, l, r);
    Node res = query(v, l, r, 1, 5);
    cout << res.index << " " << res.val << endl; // 1 15
    update(v, l, r, 3, 10);
    res = query(v, l, r, 1, 5);
    cout << res.index << " " << res.val << endl; // 1 22
    res = query(v, l, r, 3, 3);
    cout << res.index << " " << res.val << endl; // 3 10
    return 0;
}
```

### 线段树 区间修改，区间查询

```c++
int s[N];
struct Node {
    int l, r;
    int sum;
    int max;
    int min;
    int lza, lzm;
} tr[N << 2];

void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
    tr[u].max = max(tr[u << 1].max, tr[u << 1 | 1].max);
    tr[u].min = min(tr[u << 1].min, tr[u << 1 | 1].min);
}

void pushdown(int u) {
    if (tr[u].lza) {
        tr[u << 1].sum += tr[u].lza * (tr[u << 1].r - tr[u << 1].l + 1);
        tr[u << 1 | 1].sum += tr[u].lza * (tr[u << 1 | 1].r - tr[u << 1 | 1].l + 1);
        tr[u << 1].lza += tr[u].lza;
        tr[u << 1 | 1].lza += tr[u].lza;
        tr[u].lza = 0;
    }
    if (tr[u].lzm != 1) {
        tr[u << 1].sum *= tr[u].lzm;
        tr[u << 1 | 1].sum *= tr[u].lzm;
        tr[u << 1].lzm *= tr[u].lzm;
        tr[u << 1 | 1].lzm *= tr[u].lzm;
        tr[u].lzm = 1;
    }
}


void build(int u, int l, int r) {
    tr[u].l = l, tr[u].r = r;
    tr[u].lza = 0, tr[u].lzm = 1;
    if (l == r) {
        tr[u].sum = s[l];
        tr[u].max = s[l];
        tr[u].min = s[l];
        return;
    }
    int mid = (l + r) >> 1;
    build(u << 1, l, mid);
    build(u << 1 | 1, mid + 1, r);
    pushup(u);
}

void add(int u, int l, int r, int x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum += x * (tr[u].r - tr[u].l + 1);
        // update the max and min
        tr[u].max += x;
        tr[u].min += x;
        tr[u].lza += x;
        return;
    }
    pushdown(u);
    int mid = (tr[u].l + tr[u].r) >> 1;
    if (l <= mid) add(u << 1, l, r, x);
    if (r > mid) add(u << 1 | 1, l, r, x);
    pushup(u);
}

void mul(int u, int l, int r, int x) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].sum *= x;
        tr[u].max *= x;
        tr[u].min *= x;
        tr[u].lzm *= x;
        return;
    }
    pushdown(u);
    int mid = (tr[u].l + tr[u].r) >> 1;
    if (l <= mid) mul(u << 1, l, r, x);
    if (r > mid) mul(u << 1 | 1, l, r, x);
    pushup(u);
}

int query_min(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].min;
    }
    pushdown(u);
    int mid = (tr[u].l + tr[u].r) >> 1;
    int ans = 1e18;
    if (l <= mid) ans = min(ans, query_min(u << 1, l, r));
    if (r > mid) ans = min(ans, query_min(u << 1 | 1, l, r));
    return ans;
}

int query_max(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].max;
    }
    pushdown(u);
    int mid = (tr[u].l + tr[u].r) >> 1;
    int ans = 0;
    if (l <= mid) ans = max(ans, query_max(u << 1, l, r));
    if (r > mid) ans = max(ans, query_max(u << 1 | 1, l, r));
    return ans;
}

int query(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].sum;
    }
    pushdown(u);
    int mid = (tr[u].l + tr[u].r) >> 1;
    int ans = 0;
    if (l <= mid) ans += query(u << 1, l, r);
    if (r > mid) ans += query(u << 1 | 1, l, r);
    return ans;
}

signed main() {
    int n = 10;
    for (int i = 1; i <= n; i++) s[i] = i;
    build(1, 1, n);
    add(1, 1, 3, -100);
    for (int i = 1; i <= n; i++) {
        cout << query(1, i, i) << " ";
    } // -99 -98 -97 4 5 6 7 8 9 10
    cout << endl;
    mul(1, 1, 3, 2);
    for (int i = 1; i <= n; i++) {
        cout << query(1, i, i) << " ";
    } // -198 -196 -194 4 5 6 7 8 9 10
    cout << endl;
    cout << query_min(1, 1, n) << endl; // -198
    cout << query_max(1, 1, n) << endl; // 10
}
```

```c++

struct Node {
    int l, r;
    int id, val;
} tr[N << 2];

void pushup(int u) {
    if (tr[u << 1].id == tr[u << 1 | 1].id) tr[u].id = tr[u << 1].id;
    else tr[u].id = -1;
    if (tr[u << 1].val == tr[u << 1 | 1].val) tr[u].val = tr[u << 1].val;
    else tr[u].val = -1;
}

void pushdown(int u) {
    if (tr[u].l != tr[u].r) {
        if (tr[u].id != -1) tr[u << 1].id = tr[u << 1 | 1].id = tr[u].id;
        if (tr[u].val != -1) tr[u << 1].val = tr[u << 1 | 1].val = tr[u].val;
    }
}

void update(int u, int l, int r, int id, int val) {
    if (tr[u].l >= l && tr[u].r <= r) {
        tr[u].id = id;
        tr[u].val = val;
        return;
    }
    pushdown(u);
    int mid = (tr[u].l + tr[u].r) >> 1;
    if (l <= mid) update(u << 1, l, r, id, val);
    if (r > mid) update(u << 1 | 1, l, r, id, val);
    pushup(u);
}

void build(int u, int l, int r) {
    tr[u].l = l, tr[u].r = r;
    if (l == r) {
        return;
    }
    int mid = (l + r) >> 1;
    build(u << 1, l, mid);
    build(u << 1 | 1, mid + 1, r);
    pushup(u);
}

pii query(int u, int p) {
    if (tr[u].l == tr[u].r) return {tr[u].id, tr[u].val};
    pushdown(u);
    int mid = (tr[u].l + tr[u].r) >> 1;
    if (p <= mid) return query(u << 1, p);
    else return query(u << 1 | 1, p);
}

signed main() {
    int n = 10;
    build(1, 1, n);
    for (int i = 1; i <= n; i++) {
        auto [id, val] = query(1, i);
        cout << id << " " << val << " | ";
    } // 0 0 | 0 0 | 0 0 | 0 0 | 0 0 | 0 0 | 0 0 | 0 0 | 0 0 | 0 0 |
    cout << endl;
    update(1, 1, 5, 1, 1);
    for (int i = 1; i <= n; i++) {
        auto [id, val] = query(1, i);
        cout << id << " " << val << " | ";
    } // 1 1 | 1 1 | 1 1 | 1 1 | 1 1 | 0 0 | 0 0 | 0 0 | 0 0 | 0 0 |
    cout << endl;
    update(1, 3, 7, 2, 2);
    for (int i = 1; i <= n; i++) {
        auto [id, val] = query(1, i);
        cout << id << " " << val << " | ";
    } // 1 1 | 1 1 | 2 2 | 2 2 | 2 2 | 2 2 | 2 2 | 0 0 | 0 0 | 0 0 |
    cout << endl;
    update(1, 2, 3, 3, 3);
    for (int i = 1; i <= n; i++) {
        auto [id, val] = query(1, i);
        cout << id << " " << val << " | ";
    } // 1 1 | 3 3 | 3 3 | 2 2 | 2 2 | 2 2 | 2 2 | 0 0 | 0 0 | 0 0 |

}
```

### 倍增LCA

```c++
#include<cstdio>
#include<iostream>
#include<cstring>

using namespace std;
const int maxn = 500000 + 2;
int n, m, s;
int k = 0;
int head[maxn], d[maxn], p[maxn][21];
//head数组就是链接表标配了吧？d存的是深度（deep）,p[i][j]存的[i]向上走2的j次方那么长的路径
struct node {
    int v, next;
} e[maxn * 2];//存树
void add(int u, int v) {
    e[k].v = v;
    e[k].next = head[u];
    head[u] = k++;
}               //加边函数
void dfs(int u, int fa) {
    d[u] = d[fa] + 1;
    p[u][0] = fa;
    for (int i = 1; (1 << i) <= d[u]; i++)
        p[u][i] = p[p[u][i - 1]][i - 1];
    for (int i = head[u]; i != -1; i = e[i].next) {
        int v = e[i].v;
        if (v != fa) {
            dfs(v, u);
        }
    }
}                //首先进行的预处理，将所有点的deep和p的初始值dfs出来
int lca(int a, int b)                        //非常标准的lca查找
{
    if (d[a] > d[b])
        swap(a, b);           //保证a是在b结点上方，即a的深度小于b的深度
    for (int i = 20; i >= 0; i--)
        if (d[a] <= d[b] - (1 << i))
            b = p[b][i];             //先把b移到和a同一个深度
    if (a == b)
        return a;    //特判，如果b上来和就和a一样了，那就可以直接返回答案了
    for (int i = 20; i >= 0; i--) {
        if (p[a][i] == p[b][i])
            continue;
        else
            a = p[a][i], b = p[b][i];           //A和B一起上移
    }
    return p[a][0];               //找出最后a值的数字
}

int main() {
    memset(head, -1, sizeof(head));
    int a, b;
    scanf("%d%d%d", &n, &m, &s);
    for (int i = 1; i < n; i++) {
        scanf("%d%d", &a, &b);
        add(a, b);
        add(b, a);               //无向图，要加两次
    }
    dfs(s, 0);
    for (int i = 1; i <= m; i++) {
        scanf("%d%d", &a, &b);
        printf("%d\n", lca(a, b));
    }
    return 0;
}
```

### Dijkstra Bfs Dfs

```c++
#include "bits/stdc++.h"
using namespace std;
using P = pair<int,int>;
const int N = 500;
int n, m, x, y, z, head[N], path[N], dist[N], cnt;
bool used[N];
struct E {
    int to, w, next;
}edge[N * N];

inline void addEdge(int u, int v, int w) {
    edge[cnt] = {v, w, head[u]};
    head[u] = cnt++;
}

inline void dfs(int u) {
    cout << u << endl;
    used[u] = true;
    for (int i = head[u]; i != -1; i = edge[i].next) {
        int v = edge[i].to;
        if (!used[v]) {
            dfs(v);
        }
    }
}

inline void bfs(int s) {
    queue<int> q;
    q.push(s);
    while(!q.empty()) {
        int u = q.front();
        q.pop();
        if (used[u]) {
            continue;
        }
        cout << u << endl;
        used[u] = true;
        for (int i = head[u]; i != -1; i = edge[i].next) {
            q.push(edge[i].to);
        }
    }
}

inline void dij(int s) {
    memset(dist, 0x3f, sizeof dist);
    priority_queue<P, vector<P>, greater<>> pq;
    pq.push({0, s});
    dist[s] = 0;
    while(!pq.empty()) {
        auto t = pq.top();
        pq.pop();
        int u = t.second, w = t.first;
        if (used[u]) {
            continue;
        }
        used[u] = true;
        for (int i = head[u]; i != -1; i = edge[i].next) {
            int v = edge[i].to;
            if (dist[v] > w + edge[i].w) {
                dist[v] = w + edge[i].w;
                path[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
}

void print(int u) {
    if (u == 1) {
        cout << u;
        return;
    }
    print(path[u]);
    cout << " " << u;
}
int main() {
    memset(head, -1, sizeof head);
    cin >> n >> m;
    for (int i = 0;i < m; ++i) {
        cin >> x >> y >> z;
        addEdge(x, y, z);
        addEdge(y, x, z);
    }
    cout << "dfs" << endl;
    dfs(1);
    memset(used, false, sizeof used);
    cout << "bfs" << endl;
    bfs(1);
    memset(used, false, sizeof used);
    cout << "dij" << endl;
    dij(1);
    for (int i = 1;i <= n; ++i) {
        cout << i << " " << dist[i] << endl;
    }
    print(4);
    cout << endl;
    return 0;
}
// 10 11
// 8 7 2
// 6 8 6
// 4 5 3
// 8 4 7
// 8 1 4
// 1 2 5
// 1 4 1
// 9 8 5
// 9 1 8
// 1 10 3
// 2 4 4
```

### Floyd

```c++
#include<bits/stdc++.h>

using namespace std;
const int INF = 0x3f3f3f3f;

int main() {
    int e[10][10], n, m, t1, t2, t3;
    cin >> n >> m;  //n表示顶点个数，m表示边的条数
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            if (i == j)
                e[i][j] = 0;
            else
                e[i][j] = INF;
        }
    }
    for (int i = 1; i <= m; i++) {
        cin >> t1 >> t2 >> t3;
        e[t1][t2] = t3;
    }

    //核心代码
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                if (e[i][j] > e[i][k] + e[k][j])
                    e[i][j] = e[i][k] + e[k][j];
            }
        }
    }

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            printf("%3d", e[i][j]);
        }
        cout << endl;
    }
    return 0;
}

/*
4 8
1 2 2
1 3 6
1 4 4
2 3 3
3 1 7
3 4 1
4 1 5
4 3 12
*/
```

### Spfa

```c++
int head[N], ver[N], edge[N], Next[N], d[N];
// head: the head of the linked list
// ver: the vertex of the linked list
// edge: the weight of the linked list
// Next: the next node of the linked list
// d: the distance from the source node to the current node
bool vis[N];
int tot = 1;
int n, m, start;

void add(int u, int v, int w) {
    ver[++tot] = v, edge[tot] = w, Next[tot] = head[u], head[u] = tot;
}

void spfa() {
    memset(d, 0x3f, sizeof(d));
    d[start] = 0;
    queue<int> q;
    q.push(start);
    vis[start] = true;
    while (q.size()) {
        int x = q.front();
        q.pop();
        vis[x] = false;
        for (int i = head[x]; i; i = Next[i]) {
            int y = ver[i], z = edge[i];
            if (d[y] > d[x] + z) {
                d[y] = d[x] + z;
                if (!vis[y]) {
                    q.push(y);
                    vis[y] = true;
                }
            }
        }
    }
}
```

### In Pre Post

```c++
#include "bits/stdc++.h"
using namespace std;
vector<int> pre, in;
void post(int root, int start, int end) {
    if (start > end) return;
    int i = start;
    while (i < end && in[i] != pre[root]) i++;
    post(root + 1, start, i - 1);
    post(root + 1 + i - start, i + 1, end);
    cout << pre[root];
}

#include "bits/stdc++.h"
using namespace std;
vector<int> post, in;
void pre(int root, int start, int end) {
    if (start > end) return;
    int i = start;
    while (i < end && in[i] != post[root]) i++;
    cout << post[root];
    pre(root - 1 - end + i, start, i - 1);
    pre(root - 1, i + 1, end);
}
```

### Level

```c++
#include "bits/stdc++.h"

using namespace std;
vector<int> post, in;
map<int, int> level;

void pre(int root, int start, int end, int index) {
    if (start > end) return;
    int i = start;
    while (i < end && in[i] != post[root]) i++;
    level[index] = post[root];
    pre(root - 1 - (end - i), start, i - 1, 2 * index + 1);
    pre(root - 1, i + 1, end, 2 * index + 2);
}

int main() {
    int n;
    scanf("%d", &n);
    post.resize(n);
    in.resize(n);
    for (int i = 0; i < n; i++) scanf("%d", &post[i]);
    for (int i = 0; i < n; i++) scanf("%d", &in[i]);
    pre(n - 1, 0, n - 1, 0);
    auto it = level.begin();
    printf("%d", it->second);
    while (++it != level.end()) printf(" %d", it->second);
    return 0;
}
```

### 链式前向星

```c++
#include<bits/stdc++.h>

using namespace std;
const int N = 1005;//点数最大值
int n, m, cnt;//n个点, m条边

struct Edge {
    int to, w, next;//终点, 边权，同起点的上一条边的编号
} edge[N];//边集
int head[N];//head[i], 表示以i为起点的第一条边在边集数组的位置（编号）
int used[N];

inline void init() {
    memset(head, -1, sizeof(head));
    memset(used, 0, sizeof(used));
}

//加边，u起点，v终点，w边权
inline void add_edge(int u, int v, int w) {
    edge[cnt].to = v; //终点
    edge[cnt].w = w; //权值
    edge[cnt].next = head[u]; //以u为起点上一条边的编号, 也就是与这个边起点相同的上一条边的编号
    head[u] = cnt++; //更新以u为起点上一条边的编号
}

inline void dfs(int u) {
    used[u] = 1;
    for (int j = head[u]; j != -1; j = edge[j].next) {
        cout << u << " " << edge[j].to << " " << edge[j].w << endl;
        int v = edge[j].to;
//        used[u] = 1;
        if (!used[v]) {
            dfs(edge[j].to);
        }
//        used[u] = 0;
    }
}

inline void dotSet(int u) {
    cout << u << endl;
    //遍历以i为起点的边
    for (int j = head[u]; j != -1; j = edge[j].next) {
        cout << u << " " << edge[j].to << " " << edge[j].w << endl;
    }
    cout << endl;
}

int main() {
    cin >> n >> m;
    int u, v, w;
    //初始化
    init();
    //输入m条边
    for (int i = 1; i <= m; ++i) {
        cin >> u >> v >> w;
//        // 加单边
//        add_edge(u, v, w);
        // 加双向边
        add_edge(u, v, w);
        add_edge(v, u, w);
    }
    //n个起点
    for (int i = 1; i <= n; ++i) {
        dotSet(i);
    }
    for (int i = 1; i <= n; ++i) {
        dfs(i);
    }
    return 0;
}
//10 11
//8 7 1
//6 8 1
//4 5 1
//8 4 1
//8 1 1
//1 2 1
//1 4 1
//9 8 1
//9 1 1
//1 10 1
//2 4 1
```

### Kruskal

```c++
#include "bits/stdc++.h"

using namespace std;
const int N = 500;
int n, z, x, y, m;
bool flag = true;
vector<int> par;

struct Edge {
    int u, v, w;
} edge[N * N];

int find(int a) {
    return par[a] == a ? a : (par[a] = find(par[a]));
}

void unionn(int a, int b) {
    par[find(a)] = par[find(b)];
}


void init() {
    par.resize(n + 1);
    for (int i = 0; i <= n; ++i) {
        par[i] = i;
    }
}

int kruskal() {
    int cost = 0, cnt = 0;
    for (int i = 0; i < m; ++i) {
        int u = find(edge[i].u);
        int v = find(edge[i].v);
        if (u != v) {
            if (i < m - 1 && edge[i].w == edge[i + 1].w) {
                int uu = find(edge[i + 1].u);
                int vv = find(edge[i + 1].v);
                if (uu == u && vv == v || uu == v && vv == u) {
                    flag = false;
                }
            }
            cost += edge[i].w;
            cnt++;
            unionn(u, v);
        }
    }
    return cnt < n - 1 ? (cnt - n) : cost;
}

int main() {
    cin >> n >> m;
    init();
    for (int i = 0; i < m; ++i) {
        cin >> x >> y >> z;
        edge[i] = {x, y, z};
    }
    sort(edge, edge + m, [&](Edge a, Edge b) {
        return a.w < b.w;
    });
    int r = kruskal();
    if (r < 0) {
        cout << "No MST" << endl;
        // 连通图个数
        cout << -r;
    } else {
        cout << r << endl;
        // 最小生成树的唯一性
        cout << (flag ? "Yes" : "No");
    }
    return 0;
}
```

### 拓扑排序

```c++
#include "bits/stdc++.h"

using namespace std;

int nv, ne;
vector<vector<int>> g;
vector<int> vis;
vector<int> ans;

void dfs(int u) {
    vis[u] = 1;
    for (int v: g[u]) {
        if (!vis[v]) {
            dfs(v);
        }
    }
    ans.push_back(u);
}

void topological_sort() {
    vis.assign(nv, 0);
    ans.clear();
    for (int i = 0; i < nv; i++) {
        if (!vis[i]) {
            dfs(i);
        }
    }
    reverse(ans.begin(), ans.end());
}


int main() {

    cin >> nv >> ne;
    g.resize(nv);
    vis.resize(nv);
    for (int i = 0; i < ne; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        g[u].push_back(v);
    }
    topological_sort();
    for (int v: ans) {
        cout << v + 1 << ' ';
    }
    cout << '\n';
    return 0;
}
```

### 最近公共祖先

```c++
const int N = 40010, M = N * 2;
int n, m;
int h[N], e[M], ne[M], idx;
int depth[N], fa[N][16];
int q[N];

void bfs(int root) {
    memset(depth, 0x3f, sizeof depth);
    depth[0] = 0, depth[root] = 1;
    int hh = 0, tt = 0;
    q[0] = root;
    while (hh <= tt) {
        int t = q[hh++];
        for (int i = h[t]; ~i; i = ne[i]) {
            int j = e[i];
            if (depth[j] > depth[t] + 1) {
                depth[j] = depth[t] + 1;
                fa[j][0] = t;
                for (int k = 1; k <= 15; k++) {
                    fa[j][k] = fa[fa[j][k - 1]][k - 1];
                }
                q[++tt] = j;
            }
        }
    }
}

int lca(int a, int b) {
    if (depth[a] < depth[b]) swap(a, b);
    for (int k = 15; k >= 0; k--) {
        if (depth[fa[a][k]] >= depth[b]) a = fa[a][k];
    }
    if (a == b) return a;
    for (int k = 15; k >= 0; k--) {
        if (fa[a][k] != fa[b][k]) {
            a = fa[a][k];
            b = fa[b][k];
        }
    }
    return fa[a][0];
}

int main() {
    cin >> n >> m;
    memset(h, -1, sizeof h);
    for (int i = 0; i < n - 1; i++) {
        int a, b;
        cin >> a >> b; // a: parent, b: child
        e[idx] = b, ne[idx] = h[a], h[a] = idx++;
        e[idx] = a, ne[idx] = h[b], h[b] = idx++;
    }
    bfs(1);
    while (m--) {
        int a, b;
        cin >> a >> b;
        cout << lca(a, b) << endl;
    }
    return 0;
}
```

