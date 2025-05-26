def strip_code_fences(content: str) -> str:
        """
        Remove enclosing triple backticks and optional language tags if the 
        entire string is fenced. Preserves internal whitespace/indentation.
        """
        content = content.rstrip()
        if content.startswith("```") and content.endswith("```"):
            # Remove opening and closing code fences
            content = content.removeprefix("```").removesuffix("```")
            
            # Remove trailing whitespace only
            content = content.rstrip()
            
            # Check for language tags at the beginning and remove them
            for lang_tag in ("markdown", "md"):
                if content.startswith(lang_tag):
                    content = content[len(lang_tag):]
                    # Only strip leading whitespace from the language tag line, preserve content indentation
                    content = content.lstrip()
                    break
            else:
                # If no language tag is found, strip only leading newlines (\n and \r) while preserving spaces and tabs.
                content = content.lstrip('\n\r')
        return content