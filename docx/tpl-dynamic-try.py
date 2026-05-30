from docxtpl import DocxTemplate

doc = DocxTemplate('template.docx')

context = {
    'data': [
        {'col1': 'A1', 'col2': 'B1', 'col3': 'C1', 'col4': 'D1', 'col5': 'E1'},
        {'col1': 'A2', 'col2': 'B2', 'col3': 'C2', 'col4': 'D2', 'col5': 'E2'},
        {'col1': 'A3', 'col2': 'B3', 'col3': 'C3', 'col4': 'D3', 'col5': 'E3'},
    ]
}

try:
    doc.render(context)
    doc.save('output.docx')
except Exception as e:
    print(f"文档操作失败：{e}")
