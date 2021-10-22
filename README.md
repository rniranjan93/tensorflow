#include<bits/stdc++.h>
using namespace std;
#pragma warning(disable:4996)
vector<int>graph[100002];
long long sum = 0;
vector<int> cnt(100002);
vector<int> score(100002);
void dfs(int node, int parent)
{
	for (auto k:graph[node])
	{
		if (k == parent)
			continue;
		dfs(k, node);
		cnt[node] += cnt[k];
		score[node] += score[k] + cnt[k];
	}
	cnt[node]++;
}
void dfss(int node, int parent, long long s,int c)
{
	for (auto k : graph[node])
	{
		if (k == parent)
			continue;
		dfss(k, node,s+score[node]-score[k]+c-cnt[k]+1,cnt[node]-cnt[k]+c);
	}
	sum += (score[node] + s + c+cnt[node]-1)/2;
}
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	int n;
	int a, b;
	cin >> n;
	int k = -1;
	for (int i = 1; i < n; i++)
	{
		cin >> a >> b;
		graph[a].push_back(b);
		graph[b].push_back(a);
		if (graph[a].size() == 1)
			k = a;
		if (graph[b].size() == 1)
			k = b;
	}
	dfs(k,-1);
	dfss(k,-1,0,0);
	cout << (sum / 2);
	return 0;
}
