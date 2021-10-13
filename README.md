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
set<pair<long long, int>>arr;
long long values[M];
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
	int n, m, x, y;
	long long w;
	cin >> n >> m;
	for (int i = 1; i <= n; i++) {
		cin >> w;
		values[i] = w;
		arr.insert({ w,i });
	}
	pair<long long, pair<int, int>>p;
	p.first = LLONG_MAX;
	for (int i = 0; i < m; i++)
	{
		cin >> x >> y >> w;
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
	auto it = arr.begin();
	it++;
	pair<long long, int> least = *arr.begin();
	if (least.first + (*it).first > p.first)
	{
		sum += p.first;
		if (values[p.second.first] > values[p.second.second])
		{
			least.first = values[p.second.second];
			least.second = p.second.second;
		}
		else
		{
			least.first = values[p.second.first];
			least.second = p.second.first;
		}
		arr.erase({ values[p.second.first] , p.second.first});
		arr.erase({ values[p.second.second] , p.second.second});
		visited[p.second.first] = true;
		visited[p.second.second] = true;
		pushh(p.second.first);
		pushh(p.second.second);
	}
	else
	{
		arr.erase(arr.begin());
		sum += least.first + (*arr.begin()).first;
		visited[least.second] = true;
		visited[(*arr.begin()).second] = true;
		pushh(least.second);
		pushh((*arr.begin()).second);
		arr.erase(arr.begin());
	}

	while (!q.empty() && (visited[q.top().second.first] && visited[q.top().second.second]))
		q.pop();
	while (!arr.empty())
	{
		if (!q.empty() && q.top().first < least.first + (*arr.begin()).first)
		{
			sum += q.top().first;
			pair<int, int>p = q.top().second;
			if (visited[p.first])
			{
				if (least.first > values[p.second])
					least = { values[p.second],p.second };
				arr.erase({ values[p.second],p.second });
				visited[p.second] = true;
				q.pop();
				pushh(p.second);
			}
			else
			{
				if (least.first > values[p.first])
					least = { values[p.first],p.first };
				arr.erase({ values[p.first],p.first });
				visited[p.first] = true;
				q.pop();
				pushh(p.first);
			}
		}
		else
		{
			sum += least.first + (*arr.begin()).first;
			if (least.first > (*arr.begin()).first)
				least = *(arr.begin());
			visited[(*arr.begin()).second] = true;
			pushh((*arr.begin()).second);
			arr.erase(arr.begin());
		}
		while (!q.empty() && (visited[q.top().second.first] && visited[q.top().second.second]))
		{
			q.pop();
		}
	}
	cout << sum;
	return 0;
}
