using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Validation;
foreach (var path in args) {
    try {
        using var doc = WordprocessingDocument.Open(path, false);
        foreach (var ver in new[] { FileFormatVersions.Office2010, FileFormatVersions.Office2019 }) {
            var v = new OpenXmlValidator(ver);
            var errs = v.Validate(doc).ToList();
            Console.WriteLine($"== {System.IO.Path.GetFileName(path)} [{ver}] lỗi: {errs.Count}");
            foreach (var g in errs.GroupBy(e => e.Description).Take(12)) {
                var e = g.First();
                Console.WriteLine($"  x{g.Count()} | {e.Part?.Uri} | {e.Path?.XPath} | {e.Description}");
            }
        }
    } catch (Exception ex) { Console.WriteLine($"== {path}: KHÔNG MỞ ĐƯỢC: {ex.Message}"); }
}
