# -*- coding: utf-8 -*-
"""
@Time    :  2020-03-10 15:31
@Author  : Andrew --> Junhui Qi
@File    : Andrew_fix_demo.py
"""


def url_fix(domain, href):
    """规范化url"""
    href = str(href)
    if href.startswith("http"):
        pass
    elif href.startswith("//"):
        href = href[2:]
    elif href.startswith("/"):
        href = domain + href
    else:
        href = domain + "/" + href
    return href


def many_filter(lst):
    """去重url"""
    lst = [s if s.startswith("http") else f"http://{s}" for s in lst]  # 加上'http://'
    lst = [s.replace("https://", "http://") for s in lst]  # 全部规范为'http://'
    lst = [s if s[7:11] == "www." else f"{s[:7]}www.{s[7:]}" for s in lst]  # 如没有'www'则加上
    lst = [s[: len(s) - 1] if s.endswith("/") else s for s in lst]  # 尾部去掉'/'
    return list(set(lst))  # 去重


if __name__ == "__main__":
    domain = "baidu.com"
    href_lst = ['']
    print(many_filter([url_fix(domain, href) for href in href_lst]))
