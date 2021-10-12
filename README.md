#include<bits/stdc++.h>
using namespace std;
#define M 200005
#define pr  pair<long long, int>, vector<pair<long long,int>>,comp<pair<long long,int>>
template<typename T>
class comp
{
public:
	bool operator()(const T& a, const T& b)
	{
		return a.first > b.first;
	}
};
priority_queue < pr > arr;
priority_queue < pair < long long, pair<int, int>>, vector<pair<long long, pair<int, int>>>, comp<pair<long long, pair<int, int>>>>q;
bool visited[M];
vector<priority_queue <pr>> table(M);
void pushh(int j)
{
	while (!table[j].empty())
	{
		pair<long long, int> pp = table[j].top();
		q.push({ pp.first,{pp.second,j} });
		table[j].pop();
	}
}
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	long long sum = 0;
	int n, m,x,y;
	long long w;
	cin >> n >> m;
	for (int i = 1; i <= n; i++) {
		cin >> w;
		arr.push({ w,i });
	}
	pair<long long, pair<int, int>>p;
	p.first = LLONG_MAX;
	for (int i = 0; i < m; i++)
	{
		cin >> x >> y>>w;
		if (p.first > w)
		{
			p = { w,{x,y} };
		}
		table[x].push({ w,y });
		table[y].push({ w,x });
	}
	if (n == 1)
	{
		cout << 0;
		return 0;
	}
	long long temp = p.first;
	pair<long long, int> least = arr.top();
	arr.pop();
	if (least.first+arr.top().first > p.first)
	{
		sum += p.first;
		visited[p.second.first] = true;
		visited[p.second.second] = true;
		pushh(p.second.first);
		pushh(p.second.second);
	}
	else
	{
		sum += least.first + arr.top().first;
		visited[least.second] = true;
		visited[arr.top().second] = true;
		pushh(least.second);
		pushh(arr.top().second);
	}
	arr.pop();
	n -= 2;

	while (!q.empty() && (visited[q.top().second.first] && visited[q.top().second.second]))
	{
		q.pop();
	}
	while (!arr.empty() && visited[arr.top().second])
		arr.pop();
	while (n && !arr.empty())
	{
		
		if (!q.empty() && q.top().first < least.first + arr.top().first)
		{
			sum += q.top().first;
			pair<int, int>p = q.top().second;
			if (visited[p.first])
			{
				visited[p.second] = true;
				pushh(p.second);
			}
			else
			{
				visited[p.first] = true;
				pushh(p.first);
			}
			q.pop();
		}
		else
		{
			sum += least.first + arr.top().first;
			visited[arr.top().second] = true;
			pushh(arr.top().second);
			arr.pop();
		}
		while (!q.empty() && (visited[q.top().second.first] && visited[q.top().second.second]))
		{
			q.pop();
		}
		while (!arr.empty() && visited[arr.top().second])
			arr.pop();
		n--;
	}
	cout << sum;
	
	return 0;
}
