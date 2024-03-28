### STL

```c++
vector, 变长数组，倍增的思想
    size()  返回元素个数
    empty()  返回是否为空
    clear()  清空
    front()/back()
    push_back()/pop_back()
    begin()/end()
    []
    支持比较运算，按字典序

pair<int, int>
    first, 第一个元素
    second, 第二个元素
    支持比较运算，以first为第一关键字，以second为第二关键字（字典序）

string，字符串
    size()/length()  返回字符串长度
    empty()
    clear()
    substr(起始下标，(子串长度))  返回子串
    c_str()  返回字符串所在字符数组的起始地址

queue, 队列
    size()
    empty()
    push()  向队尾插入一个元素
    front()  返回队头元素
    back()  返回队尾元素
    pop()  弹出队头元素

priority_queue, 优先队列，默认是大根堆
    size()
    empty()
    push()  插入一个元素
    top()  返回堆顶元素
    pop()  弹出堆顶元素
    定义成小根堆的方式：priority_queue<int, vector<int>, greater<int>> q;

stack, 栈
    size()
    empty()
    push()  向栈顶插入一个元素
    top()  返回栈顶元素
    pop()  弹出栈顶元素

deque, 双端队列
    size()
    empty()
    clear()
    front()/back()
    push_back()/pop_back()
    push_front()/pop_front()
    begin()/end()
    []

set, map, multiset, multimap, 基于平衡二叉树（红黑树），动态维护有序序列
    size()
    empty()
    clear()
    begin()/end()
    ++, -- 返回前驱和后继，时间复杂度 O(logn)

    set/multiset
        insert()  插入一个数
        find()  查找一个数
        count()  返回某一个数的个数
        erase()
            (1) 输入是一个数x，删除所有x   O(k + logn)
            (2) 输入一个迭代器，删除这个迭代器
        lower_bound()/upper_bound()
            lower_bound(x)  返回大于等于x的最小的数的迭代器
            upper_bound(x)  返回大于x的最小的数的迭代器
    map/multimap
        insert()  插入的数是一个pair
        erase()  输入的参数是pair或者迭代器
        find()
        []  注意multimap不支持此操作。 时间复杂度是 O(logn)
        lower_bound()/upper_bound()

unordered_set, unordered_map, unordered_multiset, unordered_multimap, 哈希表
    和上面类似，增删改查的时间复杂度是 O(1)
    不支持 lower_bound()/upper_bound()， 迭代器的++，--

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

### floyd

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

