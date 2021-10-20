#include<bits/stdc++.h>
using namespace std;/*
#define MOD 1000000007
#define M 200005
pair<int, int> indexes[2002];
int arr[2002][2002];
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	/*int n, k;
	cin >> n >> k;
	int sum = 0;
	for (int i = 0; i < 2002; i++)
		indexes[i] = { -1,-1 };
	char c;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cin >> c;
			if (c == 'B')
			{
				if (indexes[i].first == -1)
				{
					indexes[i].first = j;
					indexes[i].second = j;
				}
				else
					indexes[i].second = j;
			}
		}
		if (indexes[i].first == -1)
			sum++;
	}
	int maxx = 0;
	for (int i = 0; i <= n - k; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (indexes[j].first >= i && indexes[j].second <= i + k - 1)
				arr[j][i] = 1;
		}
	}
	int temp;
	for (int i = 0; i <= n - k; i++)
	{
		temp = 0;
		for (int j = 0; j < k; j++)
		{
			temp += arr[j][i];
		}
		maxx = max(maxx, temp);
		int prev = 0;
		for (int j = k; j < n; j++)
		{
			temp += arr[j][i] - arr[j-k][i];
			maxx = max(maxx, temp);
		}
	}
	cout << sum + maxx;
	return 0;
}*/#include<bits/stdc++.h>
using namespace std;
#pragma warning(disable:4996)
vector<int>v;
class node
{
public:
	node* arr[2];
	node()
	{
		arr[1] = arr[0] = NULL;
	}
	node * single()
	{
		if (arr[0] != arr[1] && (arr[0] == NULL || arr[1] == NULL)) {
			if (arr[0])
				return arr[0];
			return arr[1];
		}
		return NULL;
	}
	bool empty()
	{
		if (arr[1] == NULL && arr[0] == NULL)
			return true;
		return false;
	}
};
void dfs(int prev, node* n, int l)
{
	if (l <= 0)
		return;
	if (prev & (1 << l))
	{
		if (!n->arr[1])
			n->arr[1] = new node();
		dfs(prev, n->arr[1], l - 1);
	}
	else
	{
		if (!n->arr[0])
			n->arr[0] = new node();
		dfs(prev, n->arr[1], l - 1);
	}
}
int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(NULL);
	int n;
	cin >> n;
	v = vector<int>(n);
	for (int i = 0; i < n; i++)
		cin >> v[i];
	sort(v.begin(), v.end(), greater<int>());
	int max_length = 0;
	int element = v[0];
	while (element)
	{
		max_length++;
		element = element >> 1;
	}
	if (max_length == 0)
	{
		cout << 0;
		return 0;
	}
	int prev = -1;
	node* head = new node;
	max_length--;
	for (int i = 0; i < n; i++)
	{
		if (prev != v[i])
		{
			prev = v[i];
			dfs(prev, head, max_length);
		}
	}
	queue<node*> q;
	queue<node*>qq;
	q.push(head);
	int number = 0;
	while (!q.empty()) {
		bool b = false;
		while (!q.empty())
		{
			node* p = q.front();
			q.pop();
			if (!p->empty()) {
				if (b == false) {
					if (p->single())
					{
						b = true;
						while (!qq.empty())qq.pop();
						qq.push(p->single());
					}
					else
					{
						qq.push(p->arr[0]);
						qq.push(p->arr[1]);
					}
				}
				else
				{
					if (p->single())
					{
						qq.push(p->single());
					}
				}
			}
		}
		if (b==false && !qq.empty())
		{
			number = (number | (1<<max_length));
		}
		max_length--;
		swap(q, qq);
	}
	cout << number;
	return 0;
}
