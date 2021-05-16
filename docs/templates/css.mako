<%!
    from pdoc.html_helpers import minify_css
%>

<%def name="mobile()" filter="minify_css">
:root {
    --highlight-color: #e82;
}
.flex {
    display: flex !important;
}

body {
    line-height: 1.5em;
}

#content {
    padding: 20px;
}

#sidebar {
    padding: 30px;
    overflow: hidden;
}

#sidebar>*:last-child {
    margin-bottom: 2cm;
}

.http-server-breadcrumbs {
    font-size: 130%;
    margin: 0 0 15px 0;
}

#footer {
    font-size: .75em;
    padding: 5px 30px;
    border-top: 1px solid #eee;
    text-align: right;
}

#footer p {
    margin: 0 0 0 1em;
    display: inline-block;
}

#footer p:last-child {
    margin-right: 30px;
}

h1,
h2,
h3,
h4,
h5 {
    font-weight: 300;
}

h1 {
    font-size: 2.5em;
    line-height: 1.1em;
}

h2 {
    font-size: 1.75em;
    margin: 1em 0 .50em 0;
}

h3 {
    font-size: 1.4em;
    margin: 25px 0 10px 0;
}

h4 {
    margin: 0;
    font-size: 105%;
}

h1:target,
h2:target,
h3:target,
h4:target,
h5:target,
h6:target {
    background: var(--highlight-color);
    padding: .2em 0;
}

a {
    color: #058;
    text-decoration: none;
    transition: color .3s ease-in-out;
}

a:hover {
    color: #e82;
}

.title code {
    font-weight: bold;
    word-break: break-all;
}

h2[id^="header-"] {
    margin-top: 2em;
}

.ident {
    color: #900;
}

pre code {
    background: #f8f8f8
}

.hljs {
    padding: 1.25rem 1.5rem;
    margin-left: -15px;
    margin-right: -15px;
    border: 1px solid #eee;
    border-radius: 6px;
    background: #282c34 !important;
    color: #9da29e !important;
}

.python {
    color: #c5c8c6 !important;
}

code {
    background: #f2f2f1;
    padding: 1px 4px;
    overflow-wrap: break-word;
    font-size: 90%;
}

h1 code {
    background: transparent
}

#http-server-module-list {
    display: flex;
    flex-flow: column;
}

#http-server-module-list div {
    display: flex;
}

#http-server-module-list dt {
    min-width: 10%;
}

#http-server-module-list p {
    margin-top: 0;
}

.toc ul,
#index {
    list-style-type: none;
    margin: 0;
    padding: 0;
}

#index code {
    background: transparent;
}

#index h3 {
    padding-bottom: .5em;
    border-bottom: 1px solid #e82;
}

#index ul {
    padding: 0;
}

#index h4 {
    margin-top: .6em;
    font-weight: bold;
}


/* Make TOC lists have 2+ columns when viewport is wide enough.
    Assuming ~20-character identifiers and ~30% wide sidebar. */

@media (min-width: 200ex) {
    #index .two-column {
        column-count: 2
    }
}

@media (min-width: 300ex) {
    #index .two-column {
        column-count: 3
    }
}

dl {
    margin-bottom: 2em;
}

dl dl:last-child {
    margin-bottom: 4em;
}

dd {
    margin: 0 0 1em 3em;
}

#header-classes+dl>dd {
    margin-bottom: 3em;
}

dd dd {
    margin-left: 2em;
}

dd p {
    margin: 10px 0;
}

.name {
    background: #eee;
    font-weight: bold;
    font-size: .85em;
    padding: 5px 10px;
    display: inline-block;
    min-width: 40%;
}

.name:hover {
    background: #e0e0e0;
}

dt:target .name {
    background: var(--highlight-color);
}

.name>span:first-child {
    white-space: nowrap;
}

.name.class>span:nth-child(2) {
    margin-left: .4em;
}

.inherited {
    color: #999;
    border-left: 5px solid #eee;
    padding-left: 1em;
}

.inheritance em {
    font-style: normal;
    font-weight: bold;
}


/* Docstrings titles, e.g. in numpydoc format */

.desc h2 {
    font-weight: 400;
    font-size: 1.25em;
}

.desc h3 {
    font-weight: 400;
    font-size: 1em;
}

.desc dt code {
    background: inherit;
    /* Don't grey-back parameters */
}

.source summary,
.git-link-div {
    color: #666;
    text-align: right;
    font-weight: 400;
    font-size: .8em;
    text-transform: uppercase;
}

.source summary>* {
    white-space: nowrap;
    cursor: pointer;
}

.git-link {
    color: inherit;
    margin-left: 1em;
}

.source pre {
    max-height: 500px;
    overflow: auto;
    margin: 0;
}

.source pre code {
    font-size: 12px;
    overflow: visible;
}

.hlist {
    list-style: none;
}

.hlist li {
    display: inline;
}

.hlist li:after {
    content: ',\2002';
}

.hlist li:last-child:after {
    content: none;
}

.hlist .hlist {
    display: inline;
    padding-left: 1em;
}

img {
    max-width: 100%;
}

td {
    padding: 0 .5em;
}

.admonition {
    padding: .1em .5em;
    margin-bottom: 1em;
}

.admonition-title {
    font-weight: bold;
}

.admonition.note,
.admonition.info,
.admonition.important {
    background: #aef;
}

.admonition.todo,
.admonition.versionadded,
.admonition.tip,
.admonition.hint {
    background: #dfd;
}

.admonition.warning,
.admonition.versionchanged,
.admonition.deprecated {
    background: #fd4;
}

.admonition.error,
.admonition.danger,
.admonition.caution {
    background: lightpink;
}

.badge {
    display: inline-block;
    padding: 0.25em 0.4em;
    font-size: 75%;
    font-weight: 700;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 0.25rem;
    transition: color 0.15s ease-in-out, background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}

@media (prefers-reduced-motion: reduce) {
    .badge {
        transition: none;
    }
}

a.badge:hover,
a.badge:focus {
    text-decoration: none;
}

.badge:empty {
    display: none;
}

.btn .badge {
    position: relative;
    top: -1px;
}

.badge-pill {
    padding-right: 0.6em;
    padding-left: 0.6em;
    border-radius: 10rem;
}

.badge-primary {
    color: #fff;
    background-color: #007bff;
}

a.badge-primary:hover,
a.badge-primary:focus {
    color: #fff;
    background-color: #0062cc;
}

a.badge-primary:focus,
a.badge-primary.focus {
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.5);
}

.badge-secondary {
    color: #fff;
    background-color: #6c757d;
}

a.badge-secondary:hover,
a.badge-secondary:focus {
    color: #fff;
    background-color: #545b62;
}

a.badge-secondary:focus,
a.badge-secondary.focus {
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(108, 117, 125, 0.5);
}

.badge-success {
    color: #fff;
    background-color: #28a745;
}

a.badge-success:hover,
a.badge-success:focus {
    color: #fff;
    background-color: #1e7e34;
}

a.badge-success:focus,
a.badge-success.focus {
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.5);
}

.badge-info {
    color: #fff;
    background-color: #17a2b8;
}

a.badge-info:hover,
a.badge-info:focus {
    color: #fff;
    background-color: #117a8b;
}

a.badge-info:focus,
a.badge-info.focus {
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(23, 162, 184, 0.5);
}

.badge-warning {
    color: #212529;
    background-color: #ffc107;
}

a.badge-warning:hover,
a.badge-warning:focus {
    color: #212529;
    background-color: #d39e00;
}

a.badge-warning:focus,
a.badge-warning.focus {
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(255, 193, 7, 0.5);
}

.badge-danger {
    color: #fff;
    background-color: #dc3545;
}

a.badge-danger:hover,
a.badge-danger:focus {
    color: #fff;
    background-color: #bd2130;
}

a.badge-danger:focus,
a.badge-danger.focus {
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.5);
}

.badge-light {
    color: #212529;
    background-color: #f8f9fa;
}

a.badge-light:hover,
a.badge-light:focus {
    color: #212529;
    background-color: #dae0e5;
}

a.badge-light:focus,
a.badge-light.focus {
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(248, 249, 250, 0.5);
}

.badge-dark {
    color: #fff;
    background-color: #343a40;
}

a.badge-dark:hover,
a.badge-dark:focus {
    color: #fff;
    background-color: #1d2124;
}

a.badge-dark:focus,
a.badge-dark.focus {
    outline: 0;
    box-shadow: 0 0 0 0.2rem rgba(52, 58, 64, 0.5);
}

.search-container {
    width: 100%;
    margin-top: 15px;
    margin-bottom: 15px;
}

#search_input {
    display: inline-block;
    width: 100%;
    height: 40px;
    padding: .375rem .75rem;
    font-size: 1rem;
    line-height: 1.5;
    color: white;
    background: #282c34 !important;
    border: none;
    border-radius: 6px;
    border-bottom: 1px solid #e82;
    outline: none;
}

.algolia-autocomplete {
    width: 100%;
    background: rgba(0, 0, 0, .2);
    border: none;
    border-radius: 6px;
}

.algolia-autocomplete input {
    display: none;
}

.index-caption {
    color: white;
}

#index a, #index h3, .toc a {
    color: white;
}

#index a:hover, .toc a:hover {
    color: #e82;
}

#sidebar {
    background: #393f4a;
}

.toc ul ul, #index ul {
    padding-left: 1.5em;
}

.toc>ul>li {
    margin-top: .5em;
}

pre {
    position: relative;
    background: #fafafa;
}

pre .btnIcon {
    position: absolute;
    top: 4px;
    z-index: 2;
    cursor: pointer;
    border: 1px solid transparent;
    padding: 0;
    color: #383a42;
    background-color: transparent;
    height: 30px;
    transition: all .25s ease-out;
}

pre .btnIcon:hover {
    text-decoration: none;
}

.btnIcon__body {
    align-items: center;
    display: flex;
    color: #abb2bf;
}

.btnIcon svg {
    fill: currentColor;
    margin-right: .4em;
}

.btnIcon__label {
    font-size: 11px;
}

.btnClipboard {
    right: 10px;
}
</%def>

<%def name="desktop()" filter="minify_css">
@media screen and (min-width: 700px) {
	#sidebar {
		width: 400px;
		height: 100vh;
		overflow: visible;
		position: sticky;
		top: 0;
	}
	#content {
        width: 100%;
		max-width: 100ch;
		padding: 3em 4em;
	}
	.item .name {
		font-size: 1em;
	}
	main {
		display: flex;
		flex-direction: row-reverse;
		justify-content: flex-end;
	}
    .scrollable-index {
        overflow-y: scroll;
        height: calc(100vh - 250px);
    }
}
</%def>

<%def name="print()" filter="minify_css">
@media print {
	#sidebar h1 {
		page-break-before: always;
	}
	.source {
		display: none;
	}
}

@media print {
	* {
		background: transparent !important;
		color: #000 !important;
		/* Black prints faster: h5bp.com/s */
		box-shadow: none !important;
		text-shadow: none !important;
	}
	a[href]:after {
		content: " (" attr(href) ")";
		font-size: 90%;
	}
	/* Internal, documentation links, recognized by having a title,
       don't need the URL explicity stated. */
	a[href][title]:after {
		content: none;
	}
	abbr[title]:after {
		content: " (" attr(title) ")";
	}
	/*
     * Don't show links for images, or javascript/internal links
     */
	.ir a:after,
	a[href^="javascript:"]:after,
	a[href^="#"]:after {
		content: "";
	}
	pre,
	blockquote {
		border: 1px solid #999;
		page-break-inside: avoid;
	}
	thead {
		display: table-header-group;
		/* h5bp.com/t */
	}
	tr,
	img {
		page-break-inside: avoid;
	}
    img {
        max-width: 100% !important;
    }
	@page {
		margin: 0.5cm;
	}
	p,
	h2,
	h3 {
		orphans: 3;
		widows: 3;
	}
	h1,
	h2,
	h3,
	h4,
	h5,
	h6 {
		page-break-after: avoid;
	}
}
</%def>
