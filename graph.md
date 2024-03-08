链式前向星

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

强连通

```c++
#include<cstdio>
#include<algorithm>
#include<cstring>

using namespace std;
struct node {
    int v, next;
} edge[1001];
int dfn[1001], low[1001];
int stack[1001], head[1001], visit[1001], cnt, tot, index;

inline void add(int x, int y) {
    edge[++cnt].next = head[x];
    edge[cnt].v = y;
    head[x] = cnt;
}

inline void tarjan(int x)//代表第几个点在处理。递归的是点。
{
    dfn[x] = low[x] = ++tot;// 新进点的初始化。
    stack[++index] = x;//进站
    visit[x] = 1;//表示在栈里
    for (int i = head[x]; i != -1; i = edge[i].next) {
        if (!dfn[edge[i].v]) {//如果没访问过
            tarjan(edge[i].v);//往下进行延伸，开始递归
            low[x] = min(low[x], low[edge[i].v]);//递归出来，比较谁是谁的儿子／父亲，就是树的对应关系，涉及到强连通分量子树最小根的事情。
        } else if (visit[edge[i].v]) {  //如果访问过，并且还在栈里。
            low[x] = min(low[x], dfn[edge[i].v]);//比较谁是谁的儿子／父亲。就是链接对应关系
        }
    }
    if (low[x] == dfn[x]) //发现是整个强连通分量子树里的最小根。
    {
        do {
            printf("%d ", stack[index]);
            visit[stack[index]] = 0;
            index--;
        } while (x != stack[index + 1]);//出栈，并且输出。
        printf("\n");
    }
}

int main() {
    memset(head, -1, sizeof(head));
    int n, m;
    scanf("%d%d", &n, &m);
    int x, y;
    for (int i = 1; i <= m; i++) {
        scanf("%d%d", &x, &y);
        add(x, y);
    }
    for (int i = 1; i <= n; i++)
        if (!dfn[i]) tarjan(1);//当这个点没有访问过，就从此点开始。防止图没走完
    return 0;
}
//6 8
//1 3
//1 2
//2 4
//3 4
//3 5
//4 6
//4 1
//5 6
```

kruskal

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

prim

```c++
#include "bits/stdc++.h"

using namespace std;
using PII = pair<int, int>;

int prim(int x, const vector<vector<PII> > &graph) {
    // priority queue to maintain edges with respect to weights
    priority_queue<PII, vector<PII>, greater<> > pq;
    vector<bool> used(graph.size(), false);
    int minimum_cost = 0;

    pq.push({0, x});
    while (!pq.empty()) {
        // Select the edge with minimum weight
        PII p = pq.top();
        pq.pop();
        x = p.second;
        // Checking for cycle
        if (used[x]) {
            continue;
        }
        minimum_cost += p.first;
        used[x] = true;
        for (const PII &neighbor: graph[x]) {
            int y = neighbor.second;
            if (!used[y]) {
                pq.push(neighbor);
            }
        }
    }
    return minimum_cost;
}

int main() {
    int nodes = 0, edges = 0;
    cin >> nodes >> edges;  // number of nodes & edges in graph
    if (nodes == 0 || edges == 0) {
        return 0;
    }

    vector<vector<PII> > graph(nodes);

    // Edges with their nodes & weight
    for (int i = 0; i < edges; ++i) {
        int x = 0, y = 0, weight = 0;
        cin >> x >> y >> weight;
        graph[x].push_back({weight, y});
        graph[y].push_back({weight, x});
    }

    // Selecting 1 as the starting node
    int minimum_cost = prim(1, graph);
    cout << minimum_cost << endl;
    return 0;
}
```

拓扑排序

```c++
#include <algorithm>
#include <iostream>
#include <vector>

int number_of_vertices,
        number_of_edges;  // For number of Vertices (V) and number of edges (E)
std::vector<std::vector<int>> graph;
std::vector<bool> visited;
std::vector<int> topological_order;

void dfs(int v) {
    visited[v] = true;
    for (int u: graph[v]) {
        if (!visited[u]) {
            dfs(u);
        }
    }
    topological_order.push_back(v);
}

void topological_sort() {
    visited.assign(number_of_vertices, false);
    topological_order.clear();
    for (int i = 0; i < number_of_vertices; ++i) {
        if (!visited[i]) {
            dfs(i);
        }
    }
    reverse(topological_order.begin(), topological_order.end());
}

int main() {
    std::cout
            << "Enter the number of vertices and the number of directed edges\n";
    std::cin >> number_of_vertices >> number_of_edges;
    int x = 0, y = 0;
    graph.resize(number_of_vertices, std::vector<int>());
    for (int i = 0; i < number_of_edges; ++i) {
        std::cin >> x >> y;
        x--, y--;  // to convert 1-indexed to 0-indexed
        graph[x].push_back(y);
    }
    topological_sort();
    std::cout << "Topological Order : \n";
    for (int v: topological_order) {
        std::cout << v + 1
                  << ' ';  // converting zero based indexing back to one based.
    }
    std::cout << '\n';
    return 0;
}

//入度为0输出
//bool topsort() {
//    int hh = 0, tt = -1;
//
//    // d[i] 存储点i的入度
//    for (int i = 1; i <= n; i++)
//        if (!d[i])
//            q[++tt] = i;
//
//    while (hh <= tt) {
//        int t = q[hh++];
//
//        for (int i = h[t]; i != -1; i = ne[i]) {
//            int j = e[i];
//            if (--d[j] == 0)
//                q[++tt] = j;
//        }
//    }
//
//    // 如果所有点都入队了，说明存在拓扑序列；否则不存在拓扑序列。
//    return tt == n - 1;
//}
```

拓扑排序

```c++
class Solution {
private:
    // 存储有向图
    vector<vector<int>> edges;
    // 标记每个节点的状态：0=未搜索，1=搜索中，2=已完成
    vector<int> visited;
    // 用数组来模拟栈，下标 0 为栈底，n-1 为栈顶
    vector<int> result;
    // 判断有向图中是否有环
    bool valid = true;

public:
    void dfs(int u) {
        // 将节点标记为「搜索中」
        visited[u] = 1;
        // 搜索其相邻节点
        // 只要发现有环，立刻停止搜索
        for (int v: edges[u]) {
            // 如果「未搜索」那么搜索相邻节点
            if (visited[v] == 0) {
                dfs(v);
                if (!valid) {
                    return;
                }
            }
                // 如果「搜索中」说明找到了环
            else if (visited[v] == 1) {
                valid = false;
                return;
            }
        }
        // 将节点标记为「已完成」
        visited[u] = 2;
        // 将节点入栈
        result.push_back(u);
    }

    vector<int> findOrder(int numCourses, vector<vector<int>> &prerequisites) {
        edges.resize(numCourses);
        visited.resize(numCourses);
        for (const auto &info: prerequisites) {
            edges[info[1]].push_back(info[0]);
        }
        // 每次挑选一个「未搜索」的节点，开始进行深度优先搜索
        for (int i = 0; i < numCourses && valid; ++i) {
            if (!visited[i]) {
                dfs(i);
            }
        }
        if (!valid) {
            return {};
        }
        // 如果没有环，那么就有拓扑排序
        // 注意下标 0 为栈底，因此需要将数组反序输出
        reverse(result.begin(), result.end());
        return result;
    }
};
```

```c++

```

```c++
#include "bits/stdc++.h"

using namespace std;
using PII = pair<int, int>;
const int N = 10010;

struct Edge {
    int to, w, next;
} edge[N * N];
int head[N], path[N], dist[N], cnt, n;
bool used[N];


inline void addEdge(int u, int v, int w) {
    edge[cnt].to = v;
    edge[cnt].w = w;
    edge[cnt].next = head[u];
    head[u] = cnt++;
}

inline void dijkstra(int s, int t) {
    memset(dist, 0x3f, sizeof dist);
    dist[s] = 0;
    priority_queue<PII, vector<PII>, greater<>> pq;
    pq.push({0, s});
    while (!pq.empty()) {
        auto p = pq.top();
        pq.pop();
        int w = p.first, u = p.second;
        if (used[u]) {
            continue;
        }
        used[u] = true;
        for (int i = head[u]; i != -1; i = edge[i].next) {
            auto j = edge[i].to;
            if (dist[j] > w + edge[i].w) {
                path[j] = u;
                dist[j] = w + edge[i].w;
                pq.push({dist[j], j});
            }
            if (j == t) {
                break;
            }
        }
    }
}

int main() {
    memset(head, -1, sizeof head);
    int m, u, v, w, s, t;
    cin >> n >> m >> s >> t;
    for (int i = 0; i < m; ++i) {
        cin >> u >> v >> w;
        addEdge(u, v, w);
        addEdge(v, u, w);
    }
    dijkstra(s, t);
    for (int i = 1; i <= n; ++i) {
        cout << dist[i] << " ";
    }
    cout << endl;
    for (int i = 3; i; i = path[i]) {
        cout << i << " ";
    }
    return 0;
}
//4 4 1 2
//1 2 2
//1 4 8
//3 2 16
//3 4 10
```

floyd

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

最长路径

```c++
#include "bits/stdc++.h"

using namespace std;
const int N = 55;
int n, e;
int graph[N][N], dp[N];
int ind[N], path[N];

int dfs(int i) {
    if (dp[i]) {
        return dp[i];
    }
    for (int j = 0; j < n; ++j) {
        if (i != j && graph[i][j]) {
            if (dp[i] < graph[i][j] + dfs(j)) {
                dp[i] = graph[i][j] + dfs(j);
                path[i] = j;
            }
        }
    }
    return dp[i];
}

void print(int i) {
    cout << i;
    while ((i = path[i]) != -1) {
        cout << "->" << i;
    }
}

int main() {
    memset(path, -1, sizeof path);
    int u, v, w;
    cin >> n >> e;
    while (e--) {
        cin >> u >> v >> w;
        graph[u][v] = w;
        ind[v] = 1;
    }
    int f;
    for (int i = 0; i < n; ++i) {
        if (!ind[i]) {
            f = i;
            break;
        }
    }
    dfs(f);
    print(f);
}
```

最长

```c++
#include<bits/stdc++.h>

using namespace std;
const int N = 200;
int n, m, cnt;

struct Edge {
    int to, w, next;
} edge[N * N * 10];
int head[N];
int dp[N], ind[N];

inline void add_edge(int u, int v, int w) {
    edge[cnt].to = v;
    edge[cnt].w = w;
    edge[cnt].next = head[u];
    head[u] = cnt++;
}

inline void solve() {
    queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (!ind[i]) {
            q.push(i);
        }
    }
    int k = 0, ans = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        k++;
        for (int i = head[u]; i != -1; i = edge[i].next) {
            int v = edge[i].to;
            ind[v]--;
            // ###########
            dp[v] = max(dp[u] + edge[i].w, dp[v]);
            ans = max(ans, dp[v]);
            if (!ind[v]) {
                q.push(v);
            }
        }
    }
    if (k == n) {
        cout << ans;
    } else {
        cout << "Impossible";
    }
}

int main() {
    memset(head, -1, sizeof(head));
    cin >> n >> m;
    int u, v, w;
    while (m--) {
        cin >> u >> v >> w;
        add_edge(u, v, w);
        ind[v]++;
    }
    solve();
    return 0;
}
```

染色法二分图

```c++
int n;      // n表示点数
int h[N], e[M], ne[M], idx;     // 邻接表存储图
int color[N];       // 表示每个点的颜色，-1表示未染色，0表示白色，1表示黑色

// 参数：u表示当前节点，c表示当前点的颜色
bool dfs(int u, int c) {
    color[u] = c;
    for (int i = h[u]; i != -1; i = ne[i]) {
        int j = e[i];
        if (color[j] == -1) {
            if (!dfs(j, !c)) return false;
        } else if (color[j] == c) return false;
    }

    return true;
}

bool check() {
    memset(color, -1, sizeof color);
    bool flag = true;
    for (int i = 1; i <= n; i++)
        if (color[i] == -1)
            if (!dfs(i, 0)) {
                flag = false;
                break;
            }
    return flag;
}
```

关键活动

```c++
#include<iostream>
#include<vector>
#include<queue>

using namespace std;
const int N = 1010;
struct Edge {
    int to, w;
};
int n, m;
int in[N], out[N], early[N], last[N];
vector<Edge> eg[N], lg[N];
vector<int> sorted;//存储拓扑序列
priority_queue<int, vector<int>, greater<int>> que;

bool TopSorted() { //拓扑排序
    int cnt = 0;
    for (int i = 1; i <= n; i++) {
        if (in[i] == 0) {
            que.push(i);
            sorted.push_back(i);
            cnt++;
        }
    }
    while (que.size()) {
        int tmp = que.top();
        que.pop();
        for (Edge x: eg[tmp]) {
            in[x.to]--;
            if (in[x.to] == 0) {
                que.push(x.to);
                sorted.push_back(x.to);
                cnt++;
            }
        }
    }
    if (cnt == n) return true;
    else return false;
}

int main() {
    cin >> n >> m;
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        eg[u].push_back({v, w});
        lg[v].push_back({u, w});
        in[v]++;
        out[u]++;
    }
    if (TopSorted()) {
        //计算earlyTime（最早开始时间）
        int maxx = 0;
        for (int p: sorted) {
            for (Edge tmp: eg[p]) {//找到p指向的结点进行更新
                if (tmp.w + early[p] > early[tmp.to]) {
                    early[tmp.to] = tmp.w + early[p];
                    maxx = max(maxx, early[tmp.to]);
                }
            }
        }
        cout << maxx << endl;
        //计算lastTime（最晚开始时间）
        int minn = 0x3f3f3f3f;
        for (int p: sorted) {
            last[p] = 0x3f3f3f3f;
            if (out[p] == 0) {
                last[p] = maxx;
            }
        }
        //拓扑序列反向遍历
        for (auto i = sorted.rbegin(); i != sorted.rend(); i++) {
            int p = *i;
            for (Edge tmp: lg[p]) {//找到指向p的结点进行更新
                if (last[p] - tmp.w < last[tmp.to]) {
                    last[tmp.to] = last[p] - tmp.w;
                }
            }
        }
        for (int i = 1; i <= n; i++) {//题目要求交接点小到大
            for (auto it = eg[i].rbegin(); it != eg[i].rend(); it++) {//与输入的顺序相反
                Edge p = *it;
                if (last[p.to] - p.w == early[i])//关键活动
                    printf("%d->%d\n", i, p.to);
            }
        }
    } else {
        cout << 0;
    }
    return 0;
}
```



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

