import re


class GbcRegExpUtils(object):
    @staticmethod
    def regExpExtractOne(key, themap, data, maxPos=-1):
        regs = themap[key]
        if regs is None:
            return list()
        for reg in regs:
            match = re.search(reg, data)
            if match:
                if maxPos != -1 and match.start() > maxPos:
                    continue
                item = match.group(0)
                clean = GbcRegExpUtils.cleanExtract(item)
                if clean == '':
                    continue
                page = GbcRegExpUtils.extractPage(data, match.start())
                return [clean, page, item, reg]
        return list()

    @staticmethod
    def regExpExtractAll(key, themap=None, data=''):
        regs = themap[key] if themap is not None else key
        if regs is None:
            return list()
        for reg in regs:
            match = re.findall(reg, data)
            if match:
                return match
        return list()

    @staticmethod
    def regExpExtractAllby(key, themap, data):
        regs = themap[key]
        if regs is None:
            return list()
        match_list = list()
        for reg in regs:
            match = re.findall(reg, data)
            if match:
                match_list.extend(match)
        return match_list

    @staticmethod
    def extractPage(data, index):
        PAGE_SEPARATOR_START = "[[["
        PAGE_SEPARATOR_END = "]]]"

        pageSentenceIndexStart = data[:index].rfind(PAGE_SEPARATOR_START)
        pageSentenceIndexEnd = data[:index].rfind(PAGE_SEPARATOR_END)
        if pageSentenceIndexStart >= 0 and pageSentenceIndexEnd >= 0:
            return int(data[(pageSentenceIndexStart + len(PAGE_SEPARATOR_START)):pageSentenceIndexEnd])
        else:
            return None

    @staticmethod
    def cleanPageMarkers(data):
        return re.sub(r'(\[\[\[.{1,3}\]\]\])', '',
                      re.sub(r'(\s\n\n\d\s?\n\n\s)', '', data))

    @staticmethod
    def extractFindall(key, themap, data):
        regs = themap[key]
        dates_found = []
        if regs is None:
            return dates_found
        for reg in regs:
            extract_item_re = re.findall(reg, data)
            for match in extract_item_re:
                if isinstance(match, str):
                    item = match
                else:
                    item = match[0]
                if item is None:
                    return ['', '']
                dates_found.append([GbcRegExpUtils.cleanExtract(item), item, reg])

        return dates_found

    @staticmethod
    def cleanExtract(data):
        if data:
            s = data.replace('\r', '')
            s = s.replace('\n', ' ')
            s = s.strip().strip(',').strip('.').strip('-')
            s = s.replace(':', '')
            s = re.sub(' +', ' ', s)
            s = re.sub('~', '', s)
            s = re.sub('·', '', s)
            s = GbcRegExpUtils.removeCharIfMultipleArray(s, ['-', '='])
            s = s.strip()
            return s
        else:
            return data

    @staticmethod
    def eagerCleaner(data):
        s = data.replace('\r', '')
        s = s.replace('\n', ' ')
        s = s.strip().strip(',').strip('.').strip('-')
        s = s.replace(':', '')
        s = re.sub(' +', ' ', s)
        s = re.sub('~', '', s)
        s = re.sub('·', '', s)
        s = s.replace('-', '')
        s = s.replace('=', '')
        s = s.strip()
        return s

    @staticmethod
    def removeCharIfMultipleArray(text, charArr):
        for char in charArr:
            text = re.sub(char + '{2,}', char, text)
        return text

    @staticmethod
    def regSpanExpExtractAllby(key, themap, data):
        regs = themap[key]
        if regs is None:
            return list()
        reg_match_list = list()
        for reg in regs:
            match = re.finditer(reg, data)
            match_list = list(match)
            for m in match_list:
                if len(m.groups()):
                    reg_match_list.append(m.span(1))
                else:
                    reg_match_list.append((m.start(), m.string))

        return reg_match_list

    @staticmethod
    def regSortExpExtractAllby(key, themap, data):
        regs = themap[key]
        if regs is None:
            return list()
        reg_match_list = list()
        for reg in regs:
            match = re.finditer(reg, data)
            match_list = list(match)
            for m in match_list:
                if len(m.groups()):
                    reg_match_list.append(m.group(1).strip())

        sorted_list = sorted(set(reg_match_list), key=lambda x: x[0])
        reg_match_list_sorted = sorted(sorted_list, key=lambda x: len(x), reverse=True)
        return [reg_match_list_sorted.pop()] if reg_match_list_sorted else list()
