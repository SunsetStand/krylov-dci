import inspect
from pyscf.fci import selected_ci, direct_spin1

funcs = [
    ('selected_ci.make_hdiag', selected_ci.make_hdiag),
    ('selected_ci.contract_2e', selected_ci.contract_2e),
    ('selected_ci.select_strs', selected_ci.select_strs),
    ('selected_ci.enlarge_space', selected_ci.enlarge_space),
    ('selected_ci.kernel', selected_ci.kernel),
    ('selected_ci.from_fci', selected_ci.from_fci),
    ('selected_ci.gen_cre_linkstr', selected_ci.gen_cre_linkstr),
    ('direct_spin1.contract_2e', direct_spin1.contract_2e),
]

for name, func in funcs:
    print(f"=== {name} ===")
    print(inspect.signature(func))
    doc = inspect.getdoc(func)
    if doc:
        lines = doc.strip().split('\n')[:10]
        print('\n'.join(lines))
    print()
