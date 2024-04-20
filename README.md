Steps to follow for setup:

<ul>

<li> Create a virtual environment 

```
python -m venv <environment name>
```
</li>
<li>
Activate virtual environment 


- Windows 

```
source <environmentname>/Scripts/activate 
```

- Linux 

```
source <environmentname>/bin/activate
```
</li>
<li>
Install Poetry 

```
pip install poetry 
```
</li>
<li>
Clone the repository 

```
git clone <url>
```
</li>
<li>
Then run 

```
poetry install 
```
this will install all the packages installed already
</li>
<li>
To add a package

```
poetry add <packagename>
```
</li>
</ul>