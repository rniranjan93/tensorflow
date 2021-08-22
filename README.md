#include<bits/stdc++.h>
using namespace std;
int n;
bool arr[5200][5200];
bool check(int step)
{
	bool b;
	for (int i = 0; i < step; i++)
	{
		for (int j = 0; j < step; j++)
		{
			b = arr[i][j];
			for(int y=i;y<n;y+=step)
				for (int x = j; x < n; x += step)
				{
					if (b != arr[y][x])
						return false;
				}
		}
	}
	return true;
}
int main()
{
	cin >> n;
	vector<vector<int>>v(n + 2);
	char c;
	int k;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n / 4; j++)
		{
			cin >> c;
			if (c >= '0' && c <= '9')
			{
				k = c - '0';
			}
			else
				if (c >= 'A')
				{
					k = c - 'A' + 10;
				}
			for(int l=j*4;l<j*4+4;l++)
				if(k & (1<<(3-(l-j*4))))
					arr[i][l]= true;
				else
					arr[i][l]=false;
		}
	}
	int limit = sqrt(n);
	set<int> s;
	for (int i = 2; i <= limit; i++)
	{
		if (n % i == 0)
		{
			s.insert(i);
			s.insert(n / i);
			for (int j = i; j * i <= n; j++)
			{
				v[j * i].push_back(i);
			}
		}
	}
	while (!s.empty())
	{
		int j = *s.rbegin();
		if (check(j))
		{
			cout << j;
			return 0;
		}
		for (auto b : v[j])
		{
			s.erase(b);
		}
		s.erase(j);
	}
	return 0;
}
/*
6
hhardh
3 2 9 11 7 1
8
hhzarwde
3 2 6 9 4 8 7 1
6
hhaarr
1 2 3 4 5 6
*/
